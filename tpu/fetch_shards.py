#!/usr/bin/env python3
"""Download a range of ClimbMix parquet shards to local disk.

nanochat's own `python -m nanochat.dataset -n N` always fetches shards 0..N-1,
which would put an identical copy on every host. A v4-64 worker has ~88 GiB
free, and the ~1200 shards needed for a 45B-token run is ~120 GiB -- so every
host holding everything does not fit.

Instead each worker takes a disjoint range, which fits, and gives the slice 8x
more unique data than mirroring would. Pairs with NanochatLoader reading only
the batch rows its own devices own.

    python3 fetch_shards.py --worker 3 --workers 8 --per-worker 200

Resumable: existing complete files are skipped, so re-running after an
interruption costs only the missing shards.
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = ("https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle"
            "/resolve/main")
MAX_SHARD = 6542
MIN_BYTES = 1 << 20   # anything smaller than 1 MiB is a truncated download


def shard_name(index):
    return f"shard_{index:05d}.parquet"


def fetch(index, out_dir, retries=4):
    name = shard_name(index)
    path = os.path.join(out_dir, name)
    if os.path.exists(path) and os.path.getsize(path) >= MIN_BYTES:
        return ("skip", index, os.path.getsize(path))

    url = f"{BASE_URL}/{name}"
    tmp = path + ".part"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "slicemon/1"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            size = os.path.getsize(tmp)
            if size < MIN_BYTES:
                raise IOError(f"short read: {size} bytes")
            os.replace(tmp, path)
            return ("ok", index, size)
        except Exception as exc:
            if attempt == retries - 1:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                return ("fail", index, str(exc))
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", type=int, required=True, help="this host's index")
    ap.add_argument("--workers", type=int, default=8, help="hosts in the slice")
    ap.add_argument("--per-worker", type=int, default=200, help="shards per host")
    ap.add_argument("--out", default=os.path.expanduser("~/climbmix"))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    start = args.worker * args.per_worker
    end = min(MAX_SHARD + 1, start + args.per_worker)
    indices = list(range(start, end))
    if not indices:
        print(f"[w{args.worker}] nothing to do (range {start}..{end})")
        return 0

    os.makedirs(args.out, exist_ok=True)
    print(f"[w{args.worker}] shards {start}..{end - 1} ({len(indices)}) -> {args.out}",
          flush=True)
    if args.dry_run:
        print(f"[w{args.worker}] dry run; first={shard_name(indices[0])} "
              f"last={shard_name(indices[-1])}")
        return 0

    done = failed = skipped = 0
    total_bytes = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {pool.submit(fetch, i, args.out): i for i in indices}
        for n, future in enumerate(as_completed(futures), 1):
            status, index, payload = future.result()
            if status == "ok":
                done += 1
                total_bytes += payload
            elif status == "skip":
                skipped += 1
                total_bytes += payload
            else:
                failed += 1
                sys.stderr.write(f"[w{args.worker}] shard {index} failed: {payload}\n")
            if n % 10 == 0 or n == len(indices):
                elapsed = time.time() - t0
                gib = total_bytes / (1 << 30)
                rate = gib / elapsed * 3600 if elapsed else 0
                print(f"[w{args.worker}] {n}/{len(indices)}  "
                      f"ok={done} skip={skipped} fail={failed}  "
                      f"{gib:.1f} GiB  {rate:.1f} GiB/h", flush=True)

    print(f"[w{args.worker}] done: {done} downloaded, {skipped} already present, "
          f"{failed} failed, {total_bytes / (1 << 30):.1f} GiB total", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
