#!/usr/bin/env bash
# Batch sweep on 2 x v6e-8, arm 0 (one slice, no DCN) against arm 1 (two slices, correct
# hierarchy), at MATCHED tokens-per-chip so the difference between them is DCN and nothing else.
#
#   ./run_v6e_sweep.sh
#
# Every run's log is pulled to results/logs/ the moment it finishes, and results/v6e_sweep.csv
# is rewritten after each pair. The allocation is spot: a preemption mid-sweep should cost the
# run in flight, not the sweep. Nothing here depends on the TPU still existing afterwards.
set -uo pipefail

ZONE="${ZONE:-europe-west4-a}"
S0="${S0:-probe-v6e-0}"
S1="${S1:-probe-v6e-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/results"
mkdir -p "$OUT/logs"

# arm0 batch / arm1 batch. arm1 spans 16 chips, arm0 spans 8, so arm1 batch is twice arm0's
# to hold tokens-per-chip equal. tokens/chip = 64 * arm1_batch.
PAIRS=("16 32" "32 64" "64 128" "128 256")

COMMON="training.tokens.len=1024 training.steps=20 training.warmup_steps=2 training.steps_for_lr=20 \
flat_tokens.filespec=synthetic_big.zarr flat_tokens.streams=2 \
flat_tokens.read_blocks_per_shuffle_buffer=8 flat_tokens.sequences_per_read_block=256 \
checkpoint_interval=100000 paths.root_working_dir=/tmp/seqax_out"

cleanup() { for n in "$S0" "$S1"; do
  gcloud compute tpus tpu-vm ssh "$n" --zone="$ZONE" --worker=0 \
    --command="pkill -f '[j]ax0435-venv' || true; mkdir -p /tmp/seqax_out" >/dev/null 2>&1
done; }

pull() {  # pull <node> <remote-log> <local-name>
  gcloud compute tpus tpu-vm scp "$1":"$2" "$OUT/logs/$3" --zone="$ZONE" --worker=0 >/dev/null 2>&1 \
    || echo "  (warning: could not pull $3 -- node may be gone)"
}

for pair in "${PAIRS[@]}"; do
  a0=${pair% *}; a1=${pair#* }
  tpc=$((64 * a1))

  echo "=== tokens/chip=$tpc  arm0 batch=$a0 (1 slice)  arm1 batch=$a1 (2 slices) ==="

  # ---- arm 0: one slice, data axis size 1, no DCN at all
  cleanup
  gcloud compute tpus tpu-vm ssh "$S0" --zone="$ZONE" --worker=0 --command="cd ~/seqax-offload && \
    MEGASCALE_NUM_SLICES=1 ~/jax0435-venv/bin/python -m train --config-name=c4_a100x8_2b \
    +paths.model_name=arm0_b$a0 mesh.d=8 mesh.t=1 training.tokens.batch=$a0 $COMMON \
    > /tmp/arm0_b$a0.log 2>&1; echo done" >/dev/null 2>&1
  pull "$S0" "/tmp/arm0_b$a0.log" "arm0_b$a0.log"
  echo "  arm0: $(grep -oE 'Profile time: [0-9.]+' "$OUT/logs/arm0_b$a0.log" 2>/dev/null || echo FAILED)"

  # ---- arm 1: both slices, correct hierarchy. No MEGASCALE override: libtpu reads
  # NUM_SLICES=2 from tpu-env metadata and the two slices form one mesh.
  cleanup
  for n in "$S0" "$S1"; do
    gcloud compute tpus tpu-vm ssh "$n" --zone="$ZONE" --worker=0 --command="cd ~/seqax-offload && \
      ~/jax0435-venv/bin/python -m train --config-name=c4_a100x8_2b \
      +paths.model_name=arm1_b$a1 mesh.d=8 mesh.t=1 mesh.dcn=data:2 training.tokens.batch=$a1 \
      $COMMON > /tmp/arm1_b$a1.log 2>&1; echo done" >/dev/null 2>&1 &
  done
  wait
  pull "$S0" "/tmp/arm1_b$a1.log" "arm1_b$a1.log"
  echo "  arm1: $(grep -oE 'Profile time: [0-9.]+' "$OUT/logs/arm1_b$a1.log" 2>/dev/null || echo FAILED)"

  python3 "$HERE/tools/parse_sweep.py" "$OUT/logs" > "$OUT/v6e_sweep.csv" 2>/dev/null || true
done

cleanup
echo
echo "=== $OUT/v6e_sweep.csv ==="
cat "$OUT/v6e_sweep.csv" 2>/dev/null
