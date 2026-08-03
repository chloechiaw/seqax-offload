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

## 7/30/2026
- ran the two slice job. report printed 1024 MiB and predicted 86 ms. step took 1599 ms.
- wrote a microbenchmark to check the wire directly, got 0.90 GB/s per chip. ran it again on v4. also 0.90. spec says 6.25 and 12.5 so this is super low....
- the batch sweep, 4 points, one slice vs two at matched tokens per chip.
  by up to 41% on the comm bound runs. 
- 540m, 1b, 2b at fixed tokens per chip to see if model size moves anything
- seq len 2048 at the same tokens per chip
- measured ici with the same probe: 22.2 GB/s.

## 7/31/2026

- spent a few hours trying to get more v6e helppp
- found the ip address quota is 8 per region, which caps how many slices you can carve
  regardless of chips
- moved everything to v4 on demand. provisions in 7 minutes every time. should have switched
  much earlier
- wrote a 5B config to check the law is not a small model artifact. since repo is llama 2 shape,  just made that bigger. 


## 8/1/2026 
- finished running v4 sweeps, v6 is not provisionable 
- making skme graphs 

## 8/2/2026 
- v4 and v6 dotn seem tk have thag mang diffedences, sinilar mem
