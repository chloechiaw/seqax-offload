#!/usr/bin/env bash
# The three things that need the v6e specifically, before it is preempted again.
#
#   1. ICI bandwidth  -- the fast-link denominator, never measured on either generation
#   2. model-size invariance -- tests the plan's claim that model size does NOT move
#      r = dcn/compute, since gradient bytes and compute both scale linearly with params
#   3. arm 2 -- the wrong hierarchy, weight gathering forced onto DCN
#
# Results land in results/ as each stage finishes. Nothing depends on the TPU afterwards.
set -uo pipefail

ZONE="${ZONE:-europe-west4-a}"
S0="${S0:-probe-v6e-0}"; S1="${S1:-probe-v6e-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/results"; mkdir -p "$OUT/logs"

cleanup() { for n in "$S0" "$S1"; do
  gcloud compute tpus tpu-vm ssh "$n" --zone="$ZONE" --worker=0 \
    --command="pkill -f '[j]ax0435-venv' || true; mkdir -p /tmp/seqax_out" >/dev/null 2>&1
done; }

COMMON="training.tokens.len=1024 training.steps=20 training.warmup_steps=2 training.steps_for_lr=20 \
flat_tokens.filespec=synthetic_big.zarr flat_tokens.streams=2 \
flat_tokens.read_blocks_per_shuffle_buffer=8 flat_tokens.sequences_per_read_block=256 \
checkpoint_interval=100000 paths.root_working_dir=/tmp/seqax_out"

# ---------------------------------------------------------------- 1. ICI bandwidth
echo "=== [1/3] ICI bandwidth (within one slice, 8 chips) ==="
cleanup
for n in "$S0" "$S1"; do
  gcloud compute tpus tpu-vm scp "$HERE/tools/bw_probe.py" "$n":~/bw_probe.py \
    --zone="$ZONE" --worker=0 >/dev/null 2>&1
done
gcloud compute tpus tpu-vm ssh "$S0" --zone="$ZONE" --worker=0 \
  --command="MEGASCALE_NUM_SLICES=1 ~/jax0435-venv/bin/python ~/bw_probe.py ici 1 > /tmp/ici.log 2>&1; echo done" >/dev/null 2>&1
gcloud compute tpus tpu-vm scp "$S0":/tmp/ici.log "$OUT/logs/bw_ici_v6e.log" --zone="$ZONE" --worker=0 >/dev/null 2>&1
grep -E "link=|buf=" "$OUT/logs/bw_ici_v6e.log" 2>/dev/null || echo "  FAILED"

# ---------------------------------------------------------------- 2. model-size invariance
echo
echo "=== [2/3] model-size invariance: does model size move r? ==="
for size in 1b 540m; do
  # matched tokens/chip = 8192, same as the 2b sweep point
  echo "--- $size ---"
  cleanup
  gcloud compute tpus tpu-vm ssh "$S0" --zone="$ZONE" --worker=0 --command="cd ~/seqax-offload && \
    MEGASCALE_NUM_SLICES=1 ~/jax0435-venv/bin/python -m train --config-name=c4_a100x8_$size \
    +paths.model_name=sz_arm0_$size mesh.d=8 mesh.t=1 training.tokens.batch=64 $COMMON \
    > /tmp/sz_arm0_$size.log 2>&1; echo done" >/dev/null 2>&1
  gcloud compute tpus tpu-vm scp "$S0":/tmp/sz_arm0_$size.log "$OUT/logs/sz_arm0_$size.log" \
    --zone="$ZONE" --worker=0 >/dev/null 2>&1

  cleanup
  for n in "$S0" "$S1"; do
    gcloud compute tpus tpu-vm ssh "$n" --zone="$ZONE" --worker=0 --command="cd ~/seqax-offload && \
      ~/jax0435-venv/bin/python -m train --config-name=c4_a100x8_$size \
      +paths.model_name=sz_arm1_$size mesh.d=8 mesh.t=1 mesh.dcn=data:2 training.tokens.batch=128 \
      $COMMON > /tmp/sz_arm1_$size.log 2>&1; echo done" >/dev/null 2>&1 &
  done
  wait
  gcloud compute tpus tpu-vm scp "$S0":/tmp/sz_arm1_$size.log "$OUT/logs/sz_arm1_$size.log" \
    --zone="$ZONE" --worker=0 >/dev/null 2>&1
  grep -E "Model params|total .* MiB" "$OUT/logs/sz_arm1_$size.log" 2>/dev/null | head -2
done

# ---------------------------------------------------------------- 3. arm 2, wrong hierarchy
echo
echo "=== [3/3] arm 2: wrong hierarchy (d spans both slices, weights gathered over DCN) ==="
cleanup
for n in "$S0" "$S1"; do
  gcloud compute tpus tpu-vm ssh "$n" --zone="$ZONE" --worker=0 --command="cd ~/seqax-offload && \
    ~/jax0435-venv/bin/python -m train --config-name=c4_a100x8_2b \
    +paths.model_name=arm2 mesh.d=16 mesh.t=1 training.tokens.batch=128 $COMMON \
    > /tmp/arm2.log 2>&1; echo done" >/dev/null 2>&1 &
done
wait
gcloud compute tpus tpu-vm scp "$S0":/tmp/arm2.log "$OUT/logs/arm2_b128.log" --zone="$ZONE" --worker=0 >/dev/null 2>&1
grep -E "COLLECTIVES|Profile time|MFU|Traceback|RESOURCE" "$OUT/logs/arm2_b128.log" 2>/dev/null | head -4 || echo "  FAILED"

cleanup
echo
echo "=== steady-state step times, all runs ==="
python3 "$HERE/tools/step_times.py" "$OUT"/logs/sz_*.log "$OUT"/logs/arm2_b128.log "$OUT"/logs/arm1_b128.log "$OUT"/logs/arm0_b64.log 2>/dev/null
