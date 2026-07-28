#!/usr/bin/env python3
"""Train nanochat's BPE tokenizer without pulling in torch.

nanochat's own scripts/tok_train.py imports torch at module scope, which we do
not want on a TPU VM -- it's a large GPU-oriented dependency tree that can
tangle with jax[tpu]. But nanochat/tokenizer.py has no such import, so the
trainer is reachable directly. This is that script minus torch, reading the
parquet shards fetch_shards.py already placed on disk.

    python3 tok_train_tpu.py --data ~/climbmix --out ~/.cache/nanochat/tokenizer

Defaults match nanochat: 2B training characters, 10k chars per document, and a
vocab of 32,768 -- which is both nanochat's default and exactly what seqax's
model configs already expect.
"""

import argparse
import os
import sys
import time

import pyarrow.parquet as pq


def text_iterator(files, max_chars, doc_cap, report_every=200_000_000):
    total = 0
    next_report = report_every
    started = time.time()
    for path in files:
        parquet = pq.ParquetFile(path)
        for rg in range(parquet.num_row_groups):
            table = parquet.read_row_group(rg, columns=["text"])
            for doc in table.column("text").to_pylist():
                if not doc:
                    continue
                if doc_cap:
                    doc = doc[:doc_cap]
                total += len(doc)
                yield doc
                if total >= next_report:
                    rate = total / max(1e-9, time.time() - started) / 1e6
                    print(f"  {total / 1e9:.2f}B / {max_chars / 1e9:.2f}B chars "
                          f"({rate:.0f} MB/s)", flush=True)
                    next_report += report_every
                if total >= max_chars:
                    print(f"  reached {total / 1e9:.2f}B chars", flush=True)
                    return


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.expanduser("~/climbmix"))
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/nanochat/tokenizer"))
    ap.add_argument("--nanochat", default=os.path.expanduser("~/nanochat"))
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--max-chars", type=int, default=2_000_000_000)
    ap.add_argument("--doc-cap", type=int, default=10_000)
    args = ap.parse_args()

    sys.path.insert(0, args.nanochat)
    from nanochat.tokenizer import RustBPETokenizer

    files = sorted(
        os.path.join(args.data, f)
        for f in os.listdir(args.data) if f.endswith(".parquet")
    )
    if not files:
        print(f"No parquet shards in {args.data}", file=sys.stderr)
        return 1

    print(f"vocab_size: {args.vocab_size:,}")
    print(f"training on up to {args.max_chars / 1e9:.1f}B chars "
          f"from {len(files)} shards", flush=True)

    started = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(
        text_iterator(files, args.max_chars, args.doc_cap), args.vocab_size
    )
    print(f"trained in {time.time() - started:.0f}s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    tokenizer.save(args.out)

    # Prove it round-trips before anything downstream depends on it.
    reloaded = RustBPETokenizer.from_directory(args.out)
    sample = "The capital of France is Paris."
    ids = reloaded.encode(sample)
    print(f"vocab: {reloaded.get_vocab_size():,}")
    print(f"roundtrip: {sample!r} -> {len(ids)} tokens -> {reloaded.decode(ids)!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
