## 7/28/2026

- issued 2 slices with tpu v4. works!!
- can we do the same with v6e? yes took 24 minutes to fill.
- first thing that confused me: v4-32 is 16 chips, not 32. the number counts
  tensorcores and v4 has two per chip. v6e-8 IS 8 chips because v6e has one core
  per chip. so "32 chips" and "16 chips" were the same machine and i'd been
  reading it wrong.
- MEGASCALE_NUM_SLICES. because the queued resource was created with
  `--node-count=2`, gcp stamps both nodes as multislice members, so a single-slice
  run blocks forever in the megascale handshake. no error, no timeout, just sits
  at 40gb rss with all 8 vfio handles open. set it to 1 and it starts in 25s.
- v6e image is ubuntu 24.04 → PEP 668, system pip refuses. need python3.12-venv
- got the v4 baseline: 556 ms/step, 34.5% mfu, 16 chips, 2b, batch 64.
- async collective fusion flags made no difference (0 dcn traffic, makes sense).
- five seqax bugs found on the way:
  - gpu-only xla flags applied on tpu
  - xla_client import removed in new jax
  - LIBTPU_INIT_ARGS assigned instead of defaulted (new libtpu rejects the fusion
    flags on v4 outright)
  - jax.tree_map gone
  - the profiler kills every multi-host run at step 3

## 7/29/2026

- built the actual thing today. mesh that knows which axis rides which network,
  typed all_reduce with `over=`, and trace-time reporting of bytes crossing dcn.
  single-slice path verified byte-identical to before, good.
- measured ici too, which nobody had ever done: 22.2 GB/s. so ici/dcn = 25x.
- exposed dcn time is flat at 1000-1190 ms across an 8x range of compute.
- the headline:
  - 8 chips, 1 slice → 149,797 tok/s
  - 16 chips, 2 slices → 139,810 tok/s
  - 0.93x the throughput
- wow v6es are hard to get

