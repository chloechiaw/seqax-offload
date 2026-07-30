# What a slice boundary costs

Measured on TPU v4 and v6e, using seqax, July 2026.

## Why seqax

seqax's whole thesis is that **every chip-to-chip transfer should be visible in the source
code**. You write `all_gather("M/d F/t -> M F/t", w)` and the string tells you what moves and
how the sharding changes. Nothing is hidden in a compiler pass.

That works inside one slice. It stops at the slice boundary — because the notation cannot
tell a fast hop from a slow one:

```python
lax.psum(x, 'd')       # 22 GB/s   (inside a slice)
lax.psum(x, 'data')    # 0.9 GB/s  (between slices)
```

Same syntax. **25x apart in cost.**

So I extended it: the mesh now knows which axes ride which network, and the compiler prints
what crosses the slow link *before a single step runs*:

```
COLLECTIVES CROSSING DCN (per chip, per step)
  train.py:443    all_reduce  over data    1024.0 MiB
  predicted at 0.9 GB/s: 1193 ms
```

Then I used it to measure what scaling out actually costs.

---

## First: how fast are the two networks, really?

Direct measurement, no model, just one collective of a known size.

| link | v4 | v6e | docs claim |
|---|---|---|---|
| inside a slice (ICI) | — | **22.2 GB/s/chip** | — |
| between slices (DCN) | **0.90 GB/s/chip** | **0.90 GB/s/chip** | 6.25 / 12.5 |

Two things fell out immediately.

**The published numbers are ~14x optimistic.** Every prediction in my plan was built on them
and every prediction was wrong.

**v4 and v6e have identical per-chip scale-out bandwidth.** The 2x generational improvement is
real — but it lives *per host*: v6e moves 7.2 GB/s per host vs v4's 3.6. v6e also puts twice as
many chips behind that host. Per chip, it cancels exactly.

---

## Sweep 1 — does the model scale inside one slice?

This is the control. Hold work-per-chip fixed and add chips; perfect scaling means step time
doesn't move.

| chips | batch | step ms | tokens/sec | vs half |
|---|---|---|---|---|
| 2 | 16 | 562 | 29,127 | – |
| 4 | 32 | 438 | 74,898 | 2.57x |
| 8 | 64 | 438 | 149,797 | **2.00x** |

**4 to 8 chips is exactly 2.00x** — and again at a second operating point. Inside a slice, on
the fast network, this model scales perfectly.

So whatever happens next is not the model's fault.

---

## Sweep 2 — v6e across a slice boundary

Three ways to use 2 slices of 8 chips:

- **A** — one slice only (8 chips, no slow network)
- **B** — two slices, weights **copied** to each, gradients summed across *(the textbook layout)*
- **C** — two slices, weights **split** across all 16

| work/chip | A step | B step | C step | A tok/s | B tok/s | C tok/s |
|---|---|---|---|---|---|---|
| 2,048 | 188 | 1375 | 1062 | 87,381 | 23,831 | 30,840 |
| 4,096 | 250 | 1312 | 1062 | 131,072 | 49,932 | 61,681 |
| 8,192 | 438 | 1438 | 1188 | 149,797 | 91,181 | 110,376 |
| 16,384 | 875 | 1875 | 1625 | 149,797 | 139,810 | 161,319 |

**Best of each:**

| | chips | tokens/sec | vs 8 chips |
|---|---|---|---|
| A — one slice | 8 | 149,797 | — |
| B — two slices, textbook | 16 | 139,810 | **0.93x** |
| C — two slices, split | 16 | 161,319 | 1.08x |

**Doubling the hardware made it slower.** The textbook layout loses outright; the other one
barely breaks even.

---

## Sweep 3 — v4 across a slice boundary

Same model, same code, 16 chips per slice instead of 8.

| work/chip | 1 slice | 2 slices | 16ch tok/s | 32ch tok/s | gain |
|---|---|---|---|---|---|
| 2,048 | 438 | 938 | 74,898 | 69,905 | 0.93x |
| 4,096 | 562 | 1062 | 116,508 | 123,362 | 1.06x |
| 8,192 | 812 | 1312 | 161,319 | 199,729 | 1.24x |
| 16,384 | 1500 | 2000 | 174,763 | 262,144 | **1.50x** |

**On v4 the second slice pays.** Same experiment, opposite answer.

---

## Sweep 4 — how the cost grows with hop count

32 v4 chips, carved two ways. Same chips, same model, only the number of slices changes.

| slices | chips/slice | gradient shard | ring factor | predicted | **measured** |
|---|---|---|---|---|---|
| 2 | 16 | 512 MiB | `2(1/2)` = 1.0 | 597 ms | **500 ms** |
| 4 | 8 | 1 GiB | `2(3/4)` = 1.5 | 1,790 ms | **1,500 ms** |

Doubling the slice count costs **3x**, not 2x — because halving chips-per-slice doubles each
chip's gradient shard *while* the ring simultaneously lengthens. The two effects multiply.

```
predicted growth = 2 x 1.5 = 3.00x
measured growth  = 1500 / 500 = 3.00x
```

Both absolute numbers land ~16% under prediction, which is the fixed ~200 ms of overlap
showing up in both.

**Throughput at 4 slices**, 8,192 tokens/chip:

| | tokens/sec | vs 8 chips |
|---|---|---|
| 8 chips, one slice | 80,660 | 1.00x |
| 32 chips, weights copied | 113,360 | 1.41x |
| 32 chips, weights split | 139,810 | 1.73x |

4x the hardware buys 1.4-1.7x.

## The rule that explains both

Let **r = interconnect time / compute time**.

Fraction of single-slice throughput you keep = **1 / (1 + r)** — matched to three decimals
across both generations, five batch sizes, two sequence lengths, and three model sizes.

**Scaling out pays if and only if r < 1.**

| | chips/slice | gradient shard | interconnect | compute | best r | result |
|---|---|---|---|---|---|---|
| v4 | 16 | 512 MiB | 500 ms | 1500 ms | **0.33** | 1.50x |
| v6e | 8 | 1024 MiB | ~1000 ms | 875 ms | **1.36** | 0.93x |

v6e loses on **both** terms:

- **half the chips per slice** → the gradient shard each chip owns is 2x bigger
- **2.7x faster compute** → 2.7x less time to hide the transfer behind

---

## The finding

> **v6e is worse at scaling out precisely because it is the better chip.**

More compute per chip and fewer chips per slice both push `r` up, and `r` is the only thing
that matters. Per-chip scale-out bandwidth did not improve between generations — it went into
serving twice as many chips per host instead.

For anyone choosing how much compute to put behind a given amount of scale-out bandwidth,
that is the tradeoff, measured on two generations of real silicon.

---

## Three more things that fell out

**Overlap is capped, not absent.** ~190 ms of the transfer hides behind compute and no more —
even with 875 ms of compute available. No XLA flag combination moved it more than 4%. The
gradient all-reduce fires *after* the whole backward pass, so there is nothing left to overlap
with. Fixing it needs bucketing, not a flag.

**When you are interconnect-bound, longer sequences are free.** Doubling sequence length at
fixed tokens-per-chip changed step time by *zero* — the extra compute vanished into the
exposed transfer window.

**The textbook rule is wrong at two slices.** "Weights on the fast link, gradients on the slow
one" assumes all weight-gathering bytes cross the boundary. They don't: a ring all-gather over
16 chips crosses it ~2 of 16 hops, so ~1/8 of the traffic pays. A 2-replica all-reduce pays on
100%. More bytes over a cheaper path wins. The real rule is about **what fraction of a
collective's bytes cross the boundary**, not which tensor it carries.

---

## Limits

Slices are small (8–16 chips) and the model is 2B — absolute numbers do not transfer, the
shape and the formula do. Everything is two slices; the 4- and 8-slice points would test
whether the last finding survives more hops, and spot capacity has not allowed it yet.
