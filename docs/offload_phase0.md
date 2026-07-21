# Optimizer-state host offload — Phase 0 (baseline & instrumentation)

Branch: `optimizer-host-offload`

## Goal of the overall project

Add a flag that parks Adam's optimizer state (`adam_mu`, `adam_nu`) in **pinned host memory**
between steps and streams it back only for the weight update — so a single TPU host (e.g. a
TRC `v4-8`) can train a **larger dense model than its HBM allows** — then characterize when the
host↔device transfer is hidden behind compute vs. when it costs throughput.

Optimizer state is the natural first target: it is ~2× the model and is touched **once per step**,
at the very end (the update), after the whole forward/backward — a wide window to overlap the copy.

## Phase 0 scope

Establish a working baseline and the instrumentation needed to measure the offload later. No
offload logic yet.

## Findings: environment / version drift

seqax pinned `jax[cpu]==0.4.26` (early 2024). Reproducing the baseline on a modern machine surfaced
several drift issues — **the mature host-offload API (memory-kind shardings / `TransferToMemoryKind`)
postdates 0.4.26, so a JAX bump is a required dependency of this work.** These are now fixed on this
branch (part "b": bump JAX + fix zarr 3.x) so the repo runs on a current stack:

| Issue | Symptom | Fix (this branch) |
|---|---|---|
| `jaxlib==0.4.26` unavailable on Python 3.11+ | pip finds only `>=0.4.34` | bumped pin to `jax[cpu]==0.4.35` (supports `pinned_host` memory kind) |
| `init_seqax.py` hardcodes GPU XLA flags | `Unknown flags in XLA_FLAGS: --xla_gpu_...` aborts on CPU | gated the GPU flags behind `SEQAX_CPU != "1"` |
| zarr 3.x API change | `Group.empty() takes 1 positional argument...` at checkpoint | pinned `zarr<3` (full zarr-3 migration = separate follow-up) |
| `graphviz` not installed | `save_hlo_svg` → `dot: command not found` | `brew install graphviz` (already in README) |

Local CPU dev/test recipe (Python 3.11+):

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt
brew install graphviz
SEQAX_CPU=1 XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  python -m train --config-name=local_test_synthetic +paths.model_name=phase0_baseline
```

> Not yet addressed: full zarr-3 support (needs reworking `save_zarr`/`load_zarr` for the new
> Group/errors/attrs API and multihost-checkpoint testing), and JAX beyond 0.4.35 (0.11+ adds
> programmable memory policies, not needed for optimizer-state offload but useful for activation
> offload later).

## Findings: functional baseline (CPU)

The full training loop runs end-to-end on CPU (8 simulated devices, `local_test_synthetic`,
2.4M params, 50 steps): forward/backward/AdamW/loss/sharding and checkpoint save+load all work.
Loss descends **7.76 → 5.20**. Every failure encountered was environment drift, not logic.

## Instrumentation added (`train.py`)

Next to the existing MFU print, the profiled step now also reports:

- **State footprint**: bytes in `weights`, `adam_mu`, `adam_nu`, and the ratio
  `optimizer_state / weights`. Backend-independent (works on CPU).
- **Peak HBM per device**: from `jax.local_devices()[0].memory_stats()["peak_bytes_in_use"]`,
  with the device memory limit. Degrades gracefully to "unavailable" on backends without
  `memory_stats` (e.g. CPU).

Empirical confirmation of the project's premise on the tiny config:

```
State footprint (global): weights=0.009 GiB, adam_mu=0.009 GiB, adam_nu=0.009 GiB
                          (optimizer state = 2.00x weights)
Peak HBM per device: unavailable (backend has no memory_stats; expected on CPU)
```

## TPU baseline runbook (to collect the real numbers on a `v4-8`)

The peak-HBM and MFU lines only carry meaning on real hardware. On a TRC `v4-8`, run the
`c4_a100x8_*` configs (or resized variants) and record the profiled-step output for each:

| Config | Params | Peak HBM / device | HBM limit | MFU | Step time | Fits? |
|---|---|---|---|---|---|---|
| 84m  | | | | | | |
| 270m | | | | | | |
| 540m | | | | | | |
| 1b   | | | | | | |
| 2b   | | | | | | |

Then **find the OOM boundary**: increase model size (and separately batch / sequence length)
until the run OOMs *without* offload. That largest-still-fits point is the Phase 3 "capacity
unlock" target — the model we expect offload to let us train that currently cannot fit.

Deliverables this baseline feeds:
- **Memory-breakdown figure** (weights vs optimizer vs activations) — from the State-footprint line + peak HBM.
- **Capacity table** (largest model with/without offload) — from the OOM boundary.
- **Throughput-tax curve** (MFU/step-time on vs off) — from the MFU + step-time lines across sizes.

## Next: Phase 1

Wire the offload behind an `optimizer_host_offload` flag: give `adam_mu`/`adam_nu` a
`memory_kind="pinned_host"` sharding at rest and bring them to device for the update. First thing
to validate is whether the memory-kind sharding survives seqax's `typed_shard_map` boundary, or
whether the transfer must be expressed at the outer `jit` boundary. Requires the JAX bump noted above.
