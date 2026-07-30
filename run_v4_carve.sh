#!/usr/bin/env bash
# The n=4 point of the carve sweep: 32 v4 chips as 4 slices of 8.
#
# Three configurations at matched work-per-chip, so the only difference is how the chips are
# carved and what crosses the slice boundaries:
#
#   local    1 slice only, 8 chips           no DCN at all - the baseline
#   copied   4 slices, weights replicated    gradients all-reduced over 4 slices
#   split    4 slices, weights sharded over all 32   weight gathering crosses boundaries
#
# At n=2, "split" beat "copied" by a flat ~250 ms. The proposed reason: a ring all-gather
# crosses the slice boundary O(1) times regardless of ring length, while an n-replica
# all-reduce pays DCN on 2(n-1)/n of its bytes and so grows with n. If that is right, going
# 2 -> 4 slices should widen the gap: copied gets worse (1.0 -> 1.5x the shard), split does
# not. If the gap narrows instead, the explanation is wrong.
set -uo pipefail

ZONE="${ZONE:-us-central2-b}"
N0=seqax-4x16-0; N1=seqax-4x16-1; N2=seqax-4x16-2; N3=seqax-4x16-3
ALL="$N0 $N1 $N2 $N3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/results"; mkdir -p "$OUT/logs"

DATA="flat_tokens.filespec=synthetic_big.zarr flat_tokens.streams=2 \
flat_tokens.read_blocks_per_shuffle_buffer=8 flat_tokens.sequences_per_read_block=256 \
checkpoint_interval=100000 paths.root_working_dir=/tmp/seqax_out training.tokens.len=1024 \
training.steps=20 training.warmup_steps=2 training.steps_for_lr=20"

cleanup() { for n in $ALL; do
  gcloud compute tpus tpu-vm ssh "$n" --zone="$ZONE" --worker=all \
    --command="pkill -f '[t]rain' || true; mkdir -p /tmp/seqax_out" >/dev/null 2>&1
done; }

# process_index 0 is not reliably worker 0 - scan for the log that actually has step lines
pull() {  # pull <node> <remote> <local>
  for w in 0 1; do
    if gcloud compute tpus tpu-vm ssh "$1" --zone="$ZONE" --worker=$w \
         --command="grep -q 'Step ' $2 2>/dev/null" >/dev/null 2>&1; then
      gcloud compute tpus tpu-vm scp "$1":"$2" "$OUT/logs/$3" --zone="$ZONE" --worker=$w >/dev/null 2>&1
      return 0
    fi
  done
  gcloud compute tpus tpu-vm scp "$1":"$2" "$OUT/logs/$3" --zone="$ZONE" --worker=0 >/dev/null 2>&1
}

# 8 chips/slice. local uses 1 slice (8 chips); the 4-slice runs use 32, so their batch is 4x
# to hold tokens-per-chip equal.
for tpc in 4096 8192; do
  bl=$(( tpc * 8 / 1024 )); bf=$(( tpc * 32 / 1024 ))
  echo "=== tokens/chip=$tpc   local batch=$bl (8 chips)   4-slice batch=$bf (32 chips) ==="

  cleanup
  gcloud compute tpus tpu-vm ssh "$N0" --zone="$ZONE" --worker=all --command="cd ~/seqax-offload && \
    MEGASCALE_NUM_SLICES=1 python3 -m train --config-name=c4_a100x8_2b +paths.model_name=c4_local_$tpc \
    mesh.d=8 mesh.t=1 training.tokens.batch=$bl $DATA > /tmp/c4_local_$tpc.log 2>&1; echo done" >/dev/null 2>&1
  pull "$N0" "/tmp/c4_local_$tpc.log" "c4_local_$tpc.log"
  echo "  local:  $(python3 "$HERE/tools/step_times.py" "$OUT/logs/c4_local_$tpc.log" 2>/dev/null | tail -1)"

  for mode in copied split; do
    if [ "$mode" = copied ]; then MESH="mesh.d=8 mesh.t=1 mesh.dcn=data:4"; else MESH="mesh.d=8 mesh.t=1 mesh.dcn=d:4"; fi
    cleanup
    for n in $ALL; do
      gcloud compute tpus tpu-vm ssh "$n" --zone="$ZONE" --worker=all --command="cd ~/seqax-offload && \
        python3 -m train --config-name=c4_a100x8_2b +paths.model_name=c4_${mode}_$tpc \
        $MESH training.tokens.batch=$bf $DATA > /tmp/c4_${mode}_$tpc.log 2>&1; echo done" >/dev/null 2>&1 &
    done
    wait
    pull "$N0" "/tmp/c4_${mode}_$tpc.log" "c4_${mode}_$tpc.log"
    echo "  ${mode}: $(python3 "$HERE/tools/step_times.py" "$OUT/logs/c4_${mode}_$tpc.log" 2>/dev/null | tail -1)"
  done
done

cleanup
echo
echo "=== n=4 carve results ==="
python3 "$HERE/tools/step_times.py" "$OUT"/logs/c4_*.log 2>/dev/null
