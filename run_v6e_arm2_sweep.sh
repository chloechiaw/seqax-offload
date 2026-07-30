#!/usr/bin/env bash
# Two experiments on the existing 2 x v6e-8 allocation. Nothing is torn down.
#
# 1. arm 2 across the batch sweep. At 8192 tok/chip the "wrong" hierarchy (FSDP stretched
#    across both slices) beat the "correct" one by 21%. The proposed mechanism is that a ring
#    all-gather crosses the slice boundary only ~2 of 16 hops, while a 2-replica all-reduce
#    pays DCN on 100% of its bytes. If that is right, arm 1's cost is FIXED (~1000 ms) while
#    arm 2's scales with per-step weight traffic, so the gap should narrow as batch grows.
#
# 2. sequence length. Every point so far moved tokens-per-chip via batch. Moving it via
#    len=2048 instead grows compute faster than linearly (attention is quadratic in length)
#    while DCN bytes are unchanged (they depend only on params). If r is all that matters,
#    these should land on the same curve as the equivalent batch point.
set -uo pipefail

ZONE="${ZONE:-europe-west4-a}"
S0="${S0:-probe-v6e-0}"; S1="${S1:-probe-v6e-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/results"; mkdir -p "$OUT/logs"

DATA="flat_tokens.filespec=synthetic_big.zarr flat_tokens.streams=2 \
flat_tokens.read_blocks_per_shuffle_buffer=8 flat_tokens.sequences_per_read_block=256 \
checkpoint_interval=100000 paths.root_working_dir=/tmp/seqax_out \
training.steps=20 training.warmup_steps=2 training.steps_for_lr=20"

cleanup() { for n in "$S0" "$S1"; do
  gcloud compute tpus tpu-vm ssh "$n" --zone="$ZONE" --worker=0 \
    --command="pkill -f '[j]ax0435-venv' || true; mkdir -p /tmp/seqax_out" >/dev/null 2>&1
done; }

# run_both <tag> <mesh-args> <batch> <len>   -- launches on both slices (16 chips)
run_both() {
  cleanup
  for n in "$S0" "$S1"; do
    gcloud compute tpus tpu-vm ssh "$n" --zone="$ZONE" --worker=0 --command="cd ~/seqax-offload && \
      ~/jax0435-venv/bin/python -m train --config-name=c4_a100x8_2b +paths.model_name=$1 \
      $2 training.tokens.batch=$3 training.tokens.len=$4 $DATA > /tmp/$1.log 2>&1; echo done" >/dev/null 2>&1 &
  done
  wait
  gcloud compute tpus tpu-vm scp "$S0":/tmp/$1.log "$OUT/logs/$1.log" --zone="$ZONE" --worker=0 >/dev/null 2>&1
  echo "    $1: $(python3 "$HERE/tools/step_times.py" "$OUT/logs/$1.log" 2>/dev/null | tail -1)"
}

# run_one <tag> <mesh-args> <batch> <len>    -- single slice (8 chips), no DCN
run_one() {
  cleanup
  gcloud compute tpus tpu-vm ssh "$S0" --zone="$ZONE" --worker=0 --command="cd ~/seqax-offload && \
    MEGASCALE_NUM_SLICES=1 ~/jax0435-venv/bin/python -m train --config-name=c4_a100x8_2b \
    +paths.model_name=$1 $2 training.tokens.batch=$3 training.tokens.len=$4 $DATA \
    > /tmp/$1.log 2>&1; echo done" >/dev/null 2>&1
  gcloud compute tpus tpu-vm scp "$S0":/tmp/$1.log "$OUT/logs/$1.log" --zone="$ZONE" --worker=0 >/dev/null 2>&1
  echo "    $1: $(python3 "$HERE/tools/step_times.py" "$OUT/logs/$1.log" 2>/dev/null | tail -1)"
}

echo "=== [1/2] arm 2 across the batch sweep (16 chips, d stretched over both slices) ==="
for b in 32 64 256; do
  echo "--- batch $b ($((64*b)) tok/chip) ---"
  run_both "arm2_b$b" "mesh.d=8 mesh.t=1 mesh.dcn=d:2" "$b" 1024
done
echo "(batch 128 already measured at 1188 ms)"

echo
echo "=== [2/2] sequence length 2048 at 8192 tok/chip ==="
echo "--- arm0, 8 chips, batch 32 ---"
run_one  "len2k_arm0_b32" "mesh.d=8 mesh.t=1"                  32 2048
echo "--- arm1, 16 chips, batch 64 ---"
run_both "len2k_arm1_b64" "mesh.d=8 mesh.t=1 mesh.dcn=data:2"  64 2048
echo "--- arm2, 16 chips, batch 64 ---"
run_both "len2k_arm2_b64" "mesh.d=8 mesh.t=1 mesh.dcn=d:2"     64 2048

cleanup
echo
echo "=== all arm2 / len2k runs ==="
python3 "$HERE/tools/step_times.py" "$OUT"/logs/arm2_b*.log "$OUT"/logs/arm2_wrong_b128.log "$OUT"/logs/len2k_*.log 2>/dev/null
