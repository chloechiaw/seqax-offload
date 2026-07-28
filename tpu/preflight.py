#!/usr/bin/env python3
"""Per-worker readiness check. Run on every host before launching a long run.

Verifies the things that fail silently or fail late: a missing package on one
host out of eight, a tokenizer that didn't copy, a shard range that overlaps a
neighbour's. Prints one line per check and exits non-zero if any fail, so
`--worker=all` output can be eyeballed in one screen.

    gcloud compute tpus tpu-vm ssh SLICE --zone=ZONE --worker=all \
      --command='python3 ~/preflight.py'
"""

import argparse
import os
import shutil
import sys
import urllib.request

ROUNDTRIP = "The capital of France is Paris."


def worker_id():
    req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/attributes/"
        "agent-worker-number",
        headers={"Metadata-Flavor": "Google"})
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return int(resp.read().decode().strip())
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/climbmix"))
    ap.add_argument("--tokenizer", default=os.path.expanduser("~/.cache/nanochat/tokenizer"))
    ap.add_argument("--nanochat", default=os.path.expanduser("~/nanochat"))
    ap.add_argument("--min-free-gib", type=float, default=20.0)
    args = ap.parse_args()

    w = worker_id()
    tag = f"[w{w}]"
    problems = []

    def check(name, ok, detail):
        if not ok:
            problems.append(name)
        print(f"{tag} {'ok ' if ok else 'FAIL'} {name:12} {detail}", flush=True)

    # packages
    versions = {}
    for pkg in ("jax", "tiktoken", "rustbpe", "pyarrow", "zarr", "hydra"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "?")
        except Exception as exc:
            versions[pkg] = f"MISSING ({type(exc).__name__})"
    missing = [k for k, v in versions.items() if str(v).startswith("MISSING")]
    check("packages", not missing,
          " ".join(f"{k}={v}" for k, v in versions.items()))

    # shard range
    try:
        shards = sorted(f for f in os.listdir(args.data) if f.endswith(".parquet"))
    except Exception:
        shards = []
    span = f"{shards[0]}..{shards[-1]}" if shards else "none"
    check("shards", len(shards) > 0, f"{len(shards)} files  {span}")

    # disk headroom
    free_gib = shutil.disk_usage("/").free / (1 << 30)
    check("disk", free_gib >= args.min_free_gib, f"{free_gib:.0f} GiB free")

    # tokenizer, loaded exactly the way NanochatLoader loads it
    sys.path.insert(0, args.nanochat)
    try:
        from nanochat.tokenizer import RustBPETokenizer
        tok = RustBPETokenizer.from_directory(args.tokenizer)
        ids = tok.encode(ROUNDTRIP)
        vocab = tok.get_vocab_size()
        ok = tok.decode(ids) == ROUNDTRIP
        check("tokenizer", ok, f"vocab={vocab:,} {len(ids)} tokens roundtrip={ok}")
    except Exception as exc:
        check("tokenizer", False, f"{type(exc).__name__}: {exc}")

    # seqax at the expected commit, with today's loader present
    try:
        sys.path.insert(0, os.path.expanduser("~/seqax-offload"))
        import input_loader
        has = hasattr(input_loader, "NanochatDataParams")
        check("seqax", has, f"NanochatDataParams present={has}")
    except Exception as exc:
        check("seqax", False, f"{type(exc).__name__}: {exc}")

    print(f"{tag} {'READY' if not problems else 'NOT READY: ' + ','.join(problems)}",
          flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
