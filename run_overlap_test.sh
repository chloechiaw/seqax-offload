#!/usr/bin/env bash
# Can the cross-slice gradient all-reduce be overlapped with compute?
#
# The sweep showed throughput tracking 1/(1+r), meaning the DCN transfer runs strictly in
# series with compute. init_seqax.py sets flags intended to overlap collectives -- but look
# at what they actually name:
#
#   --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true
#   --xla_enable_async_all_gather=true
#
# all_GATHER, twice. Nothing about all_REDUCE. Weight gathering on ICI is an all-gather, so
# these flags were tuned for the single-slice case. The collective that crosses DCN is an
# all-reduce, and nothing here asks for it to be made async. That is the hypothesis.
#
# One arm-1 config, four flag sets. Unknown-flag failures are expected for the speculative
# ones and are recorded rather than fatal.
set -uo pipefail

ZONE="${ZONE:-europe-west4-a}"
S0="${S0:-probe-v6e-0}"; S1="${S1:-probe-v6e-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/results"; mkdir -p "$OUT/logs"

BASE="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
NOFUSE="${BASE//=true/=false}"

declare -a NAMES=(baseline nofusion fuse_all_reduce async_all_reduce)
declare -a FLAGS=(
  "$BASE"
  "$NOFUSE"
  "$BASE --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true"
  "$BASE --xla_enable_async_all_reduce=true"
)

# batch 128 over 16 chips = 8192 tokens/chip. Comfortably inside HBM (19.1 GiB observed) and
# compute is 457 ms against 1193 ms of DCN, so a working overlap has 457 ms to hide and the
# step should fall from ~1860 toward ~1193.
CFG="--config-name=c4_a100x8_2b mesh.d=8 mesh.t=1 mesh.dcn=data:2 training.tokens.batch=128 \
training.tokens.len=1024 training.steps=20 training.warmup_steps=2 training.steps_for_lr=20 \
flat_tokens.filespec=synthetic_big.zarr flat_tokens.streams=2 \
flat_tokens.read_blocks_per_shuffle_buffer=8 flat_tokens.sequences_per_read_block=256 \
checkpoint_interval=100000 paths.root_working_dir=/tmp/seqax_out"

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"; flags="${FLAGS[$i]}"
  echo "=== $name ==="
  for n in "$S0" "$S1"; do
    gcloud compute tpus tpu-vm ssh "$n" --zone="$ZONE" --worker=0 \
      --command="pkill -f '[j]ax0435-venv' || true; mkdir -p /tmp/seqax_out" >/dev/null 2>&1
  done
  for n in "$S0" "$S1"; do
    gcloud compute tpus tpu-vm ssh "$n" --zone="$ZONE" --worker=0 \
      --command="cd ~/seqax-offload && LIBTPU_INIT_ARGS='$flags' ~/jax0435-venv/bin/python -m train \
        $CFG +paths.model_name=ovl_$name > /tmp/ovl_$name.log 2>&1; echo done" >/dev/null 2>&1 &
  done
  wait
  gcloud compute tpus tpu-vm scp "$S0":"/tmp/ovl_$name.log" "$OUT/logs/overlap_$name.log" \
    --zone="$ZONE" --worker=0 >/dev/null 2>&1 || echo "  (could not pull log)"
  t=$(grep -oE 'Profile time: [0-9.]+' "$OUT/logs/overlap_$name.log" 2>/dev/null | grep -oE '[0-9.]+')
  if [ -n "$t" ]; then
    echo "  step = $(python3 -c "print(f'{float('$t')/2*1000:.1f}')") ms"
  else
    echo "  FAILED: $(grep -oE 'Unknown flags.*|F[0-9]{4}.*' "$OUT/logs/overlap_$name.log" 2>/dev/null | head -1)"
  fi
done

echo
echo "reference (from the sweep, same config): arm0=456.6 ms compute, arm1=1859.5 ms, DCN=1193 ms"
echo "perfect overlap would be max(457,1193)=1193 ms; zero overlap is 457+1193=1650 ms"
