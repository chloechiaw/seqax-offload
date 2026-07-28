#!/usr/bin/env python3
"""Measure the two numbers that decide whether the run is sound.

1. bytes/token on the real corpus with the real tokenizer. Cross-entropy is not
   comparable across tokenizers, so converting someone else's target loss (e.g.
   jaxformer's 3.28 on cl100k) into ours requires this number, not a guess.

2. Token utilization under BOS-aligned packing at a given sequence length.
   Documents that overrun the remaining row are cropped, and the wasted fraction
   determines how much raw corpus a token budget actually needs.

    python3 corpus_stats.py --docs 20000 --lens 1024 2048
"""

import argparse
import os
import sys

import pyarrow.parquet as pq


def sample_docs(data_dir, limit):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
    out = []
    for name in files:
        parquet = pq.ParquetFile(os.path.join(data_dir, name))
        for rg in range(parquet.num_row_groups):
            for doc in parquet.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                if doc:
                    out.append(doc)
                    if len(out) >= limit:
                        return out
    return out


def pack_utilization(doc_lens, seq_len):
    """Replicate NanochatLoader._fill packing and report the kept fraction."""
    used = cropped = 0
    pos = 0
    for n in doc_lens:
        room = seq_len - (pos % seq_len)
        take = min(n, room)
        used += take
        cropped += n - take
        pos += take
    total = used + cropped
    return (used / total if total else 0.0), cropped / max(1, len(doc_lens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/climbmix"))
    ap.add_argument("--tokenizer", default=os.path.expanduser("~/.cache/nanochat/tokenizer"))
    ap.add_argument("--nanochat", default=os.path.expanduser("~/nanochat"))
    ap.add_argument("--docs", type=int, default=20000)
    ap.add_argument("--lens", type=int, nargs="+", default=[1024, 2048])
    args = ap.parse_args()

    sys.path.insert(0, args.nanochat)
    from nanochat.tokenizer import RustBPETokenizer

    tok = RustBPETokenizer.from_directory(args.tokenizer)
    docs = sample_docs(args.data, args.docs)
    print(f"sampled {len(docs):,} documents")

    encoded = tok.encode(docs, num_threads=16)
    if encoded and isinstance(encoded[0], int):
        encoded = [encoded]

    doc_lens = [len(e) + 1 for e in encoded]        # +1 for the BOS we prepend
    total_tokens = sum(doc_lens)
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    total_chars = sum(len(d) for d in docs)

    print(f"tokens          {total_tokens:,}")
    print(f"bytes/token     {total_bytes / total_tokens:.3f}")
    print(f"chars/token     {total_chars / total_tokens:.3f}")
    srt = sorted(doc_lens)
    print(f"doc tokens      p50={srt[len(srt)//2]:,}  "
          f"p90={srt[int(len(srt)*.9)]:,}  max={srt[-1]:,}  "
          f"mean={total_tokens/len(srt):,.0f}")

    print("\nsequence length -> token utilization under BOS-aligned packing")
    for seq_len in args.lens:
        keep, _ = pack_utilization(doc_lens, seq_len)
        print(f"  T={seq_len:<6} keep {keep * 100:5.1f}%   waste {100 - keep * 100:5.1f}%")

    # What a 45B-token budget costs in raw corpus, and what we have.
    bpt = total_bytes / total_tokens
    print("\nbudget check (1,600 shards x ~250M chars):")
    raw_tokens = 1600 * 250e6 / (total_chars / total_tokens)
    print(f"  raw corpus    ~{raw_tokens / 1e9:.0f}B tokens")
    for seq_len in args.lens:
        keep, _ = pack_utilization(doc_lens, seq_len)
        print(f"  T={seq_len:<6} usable ~{raw_tokens * keep / 1e9:.0f}B tokens "
              f"({raw_tokens * keep / 45e9:.2f}x a 45B budget)")

    print("\nloss-target conversion (bits/byte is the tokenizer-invariant unit):")
    import math
    for name, ref_loss, ref_bpt in (("jaxformer 3.28 @ cl100k", 3.28, 4.0),):
        bpb = ref_loss / (math.log(2) * ref_bpt)
        ours = bpb * math.log(2) * bpt
        print(f"  {name}: {bpb:.3f} bits/byte -> {ours:.2f} nats on our tokenizer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
