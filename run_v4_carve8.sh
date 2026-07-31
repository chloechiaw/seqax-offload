#!/usr/bin/env bash
# The n=8 point: 32 v4 chips as 8 slices of 4. The end of the carve sweep -- below 4 chips
# per slice there is essentially no in-slice topology left to shard over.
#
# Extends the law that predicted n=4 exactly:
#   bytes crossing per chip = (8 GiB / chips_per_slice) x 2(n-1)/n
#   n=2  512 MiB x 1.00 = 512 MiB   -> measured  500 ms exposed
#   n=4    1 GiB x 1.50 = 1.5 GiB   -> measured 1500 ms exposed   (3.00x, predicted 3.00x)
#   n=8    2 GiB x 1.75 = 3.5 GiB   -> predicts ~3500 ms exposed  (2.33x)
#
# The growth factor should DROP from 3.00x to 2.33x, because 2(n-1)/n is asymptoting at 2
# while the shard keeps doubling. That bend is the thing worth confirming.
set -uo pipefail

ZONE="${ZONE:-us-central2-b}"
ALL="seqax-8x8-0 seqax-8x8-1 seqax-8x8-2 seqax-8x8-3 seqax-8x8-4 seqax-8x8-5 seqax-8x8-6 seqax-8x8-7"
N0=seqax-8x8-0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/results"; mkdir -p "$OUT/logs"

DATA="flat_tokens.filespec=synthetic_big.zarr flat_tokens.streams=2 \
flat_tokens.read_blocks_per_shuffle_buffer=8 flat_tokens.sequences_per_read_block=256 \
checkpoint_interval=100000 paths.root_working_dir=/tmp/seqax_out training.tokens.len=1024 \
training.steps=20 training.warmup_steps=2 training.steps_for_lr=20"

cleanup() { for n in $ALL; do
  gcloud compute tpus tpu-vm ssh "$n" --zone="$ZONE" --worker=all \
    --command="pkill -f '[t]rain' || true; mkdir -p /tmp/seqax_out" >/dev/null 2>&1 &
done; wait; }

pull() { gcloud compute tpus tpu-vm scp "$1":"$2" "$OUT/logs/$3" --zone="$ZONE" --worker=0 >/dev/null 2>&1; }

# 4 chips/slice. local = 1 slice (4 chips); the 8-slice runs use all 32, so 8x the batch.
for tpc in 4096 8192; do
  bl=$(( tpc * 4 / 1024 )); bf=$(( tpc * 32 / 1024 ))
  echo "=== tokens/chip=$tpc   local batch=$bl (4 chips)   8-slice batch=$bf (32 chips) ==="

  cleanup
  gcloud compute tpus tpu-vm ssh "$N0" --zone="$ZONE" --worker=0 --command="cd ~/seqax-offload && \
    MEGASCALE_NUM_SLICES=1 python3 -m train --config-name=c4_a100x8_2b +paths.model_name=c8_local_$tpc \
    mesh.d=4 mesh.t=1 training.tokens.batch=$bl $DATA > /tmp/c8_local_$tpc.log 2>&1; echo done" >/dev/null 2>&1
  pull "$N0" "/tmp/c8_local_$tpc.log" "c8_local_$tpc.log"
  echo "  local:  $(python3 "$HERE/tools/step_times.py" "$OUT/logs/c8_local_$tpc.log" 2>/dev/null | tail -1)"

  for mode in copied split; do
    if [ "$mode" = copied ]; then MESH="mesh.d=4 mesh.t=1 mesh.dcn=data:8"; else MESH="mesh.d=4 mesh.t=1 mesh.dcn=d:8"; fi
    cleanup
    for n in $ALL; do
      gcloud compute tpus tpu-vm ssh "$n" --zone="$ZONE" --worker=0 --command="cd ~/seqax-offload && \
        python3 -m train --config-name=c4_a100x8_2b +paths.model_name=c8_${mode}_$tpc \
        $MESH training.tokens.batch=$bf $DATA > /tmp/c8_${mode}_$tpc.log 2>&1; echo done" >/dev/null 2>&1 &
    done
    wait
    pull "$N0" "/tmp/c8_${mode}_$tpc.log" "c8_${mode}_$tpc.log"
    echo "  ${mode}: $(python3 "$HERE/tools/step_times.py" "$OUT/logs/c8_${mode}_$tpc.log" 2>/dev/null | tail -1)"
  done
done

cleanup
echo
echo "=== n=8 carve results ==="
python3 "$HERE/tools/step_times.py" "$OUT"/logs/c8_*.log 2>/dev/null
