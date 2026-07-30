#!/usr/bin/env bash
# Within-slice scaling on v6e: 2, 4, 8 chips, all on ONE host, no DCN anywhere.
#
# This is the control the whole DCN result needs. We know 8->16 chips ACROSS slices gives
# 0.93x - worse than useless. But that only implicates the interconnect if the model scales
# fine on the fast network. If within-slice scaling is also poor, the model is simply too
# small and DCN is a red herring.
#
# Only possible on a single-host slice: v6e-8 is 1 host with 8 chips, so a mesh over a device
# subset works. On v4 it does not - a v4-32 slice is 4 hosts and the excluded hosts fail with
# "Device assignment (Computations: N Replicas: 1...)". Verified both ways.
#
# tokens-per-chip is held FIXED within each line, so batch scales with chip count. Perfect
# scaling would show identical step time across chip counts at the same tokens/chip.
set -uo pipefail

ZONE="${ZONE:-europe-west4-a}"
S0="${S0:-probe-v6e-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/results"; mkdir -p "$OUT/logs"

COMMON="training.tokens.len=1024 training.steps=20 training.warmup_steps=2 training.steps_for_lr=20 \
flat_tokens.filespec=synthetic_big.zarr flat_tokens.streams=2 \
flat_tokens.read_blocks_per_shuffle_buffer=8 flat_tokens.sequences_per_read_block=256 \
checkpoint_interval=100000 paths.root_working_dir=/tmp/seqax_out"

# ship current code (build_mesh now accepts a device subset)
tar czf /tmp/v6w.tgz -C "$HERE/.." --exclude='__pycache__' --exclude='outputs' --exclude='.git' \
  --exclude='synthetic_big.zarr' --exclude='results' "$(basename "$HERE")"
gcloud compute tpus tpu-vm scp /tmp/v6w.tgz "$S0":~/seqax.tgz --zone="$ZONE" --worker=0 >/dev/null 2>&1
gcloud compute tpus tpu-vm ssh "$S0" --zone="$ZONE" --worker=0 \
  --command="tar xzf ~/seqax.tgz; mkdir -p /tmp/seqax_out" >/dev/null 2>&1

for tpc in 4096 8192; do
  for chips in 2 4 8; do
    batch=$(( tpc * chips / 1024 ))
    tag="w${chips}c_${tpc}"
    echo "--- chips=$chips  tokens/chip=$tpc  batch=$batch ---"
    gcloud compute tpus tpu-vm ssh "$S0" --zone="$ZONE" --worker=0 \
      --command="pkill -f '[j]ax0435-venv' || true" >/dev/null 2>&1
    gcloud compute tpus tpu-vm ssh "$S0" --zone="$ZONE" --worker=0 --command="cd ~/seqax-offload && \
      MEGASCALE_NUM_SLICES=1 ~/jax0435-venv/bin/python -m train --config-name=c4_a100x8_2b \
      +paths.model_name=$tag mesh.d=$chips mesh.t=1 training.tokens.batch=$batch $COMMON \
      > /tmp/$tag.log 2>&1; echo done" >/dev/null 2>&1
    gcloud compute tpus tpu-vm scp "$S0":/tmp/$tag.log "$OUT/logs/$tag.log" \
      --zone="$ZONE" --worker=0 >/dev/null 2>&1 || echo "  (log pull failed - node may be gone)"
    echo "    $(python3 "$HERE/tools/step_times.py" "$OUT/logs/$tag.log" 2>/dev/null | tail -1)"
  done
done

gcloud compute tpus tpu-vm ssh "$S0" --zone="$ZONE" --worker=0 \
  --command="pkill -f '[j]ax0435-venv' || true" >/dev/null 2>&1
echo
echo "=== within-slice scaling, v6e (no DCN) ==="
python3 "$HERE/tools/step_times.py" "$OUT"/logs/w*c_*.log 2>/dev/null
