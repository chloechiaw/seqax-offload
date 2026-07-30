#!/usr/bin/env bash
# v4 scaling at the slice boundary: 16 chips (one slice, ICI only) vs 32 chips (two slices,
# adds DCN), at matched tokens-per-chip so the difference is the interconnect and nothing else.
#
# This is the v4 counterpart of run_v6e_sweep.sh, and the doubling the v6e could not show:
# 8 chips within a slice is unreachable on v4, because a v4-32 slice spans 4 hosts and a mesh
# over a device subset fails on the hosts left out (verified: INVALID_ARGUMENT, device
# assignment). The smallest unit here is the whole slice.
#
# Stage 1 sets up seqax-ms-1, which has never been provisioned. Stage 2 sweeps.
# Every log is pulled locally as it completes.
set -uo pipefail

ZONE="${ZONE:-us-central2-b}"
S0="${S0:-seqax-ms-0}"; S1="${S1:-seqax-ms-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/results"; mkdir -p "$OUT/logs"

COMMON="training.tokens.len=1024 training.steps=20 training.warmup_steps=2 training.steps_for_lr=20 \
flat_tokens.filespec=synthetic_big.zarr flat_tokens.streams=2 \
flat_tokens.read_blocks_per_shuffle_buffer=8 flat_tokens.sequences_per_read_block=256 \
checkpoint_interval=100000 paths.root_working_dir=/tmp/seqax_out"

cleanup() { for n in "$S0" "$S1"; do
  gcloud compute tpus tpu-vm ssh "$n" --zone="$ZONE" --worker=all \
    --command="pkill -f '[t]rain' || true; mkdir -p /tmp/seqax_out" >/dev/null 2>&1
done; }

# process_index 0 is NOT reliably worker 0 -- on this slice it landed on worker 3. Scan for
# the log that actually has step lines in it.
pull() {  # pull <node> <remote> <local>
  for w in 0 1 2 3; do
    if gcloud compute tpus tpu-vm ssh "$1" --zone="$ZONE" --worker=$w \
         --command="grep -q 'Step ' $2 2>/dev/null" >/dev/null 2>&1; then
      gcloud compute tpus tpu-vm scp "$1":"$2" "$OUT/logs/$3" --zone="$ZONE" --worker=$w >/dev/null 2>&1
      return 0
    fi
  done
  # nothing had step lines -- keep worker 0's copy so the failure is inspectable
  gcloud compute tpus tpu-vm scp "$1":"$2" "$OUT/logs/$3" --zone="$ZONE" --worker=0 >/dev/null 2>&1
}

echo "=== stage 1: provision $S1 (repo, deps, dataset) ==="
tar czf /tmp/seqax_v4.tgz -C "$HERE/.." --exclude='__pycache__' --exclude='outputs' \
  --exclude='.git' --exclude='synthetic_big.zarr' --exclude='results' "$(basename "$HERE")"
gcloud compute tpus tpu-vm scp /tmp/seqax_v4.tgz "$S1":~/seqax.tgz --zone="$ZONE" --worker=all >/dev/null 2>&1
gcloud compute tpus tpu-vm ssh "$S1" --zone="$ZONE" --worker=all --command="
  tar xzf ~/seqax.tgz && mkdir -p /tmp/seqax_out
  pip install -q --user 'zarr<3' 'fsspec[gcs]' einops hydra-core numcodecs 'jaxtyping==0.2.34' 2>&1 | tail -1
  cd ~/seqax-offload/tools && rm -rf ~/seqax-offload/synthetic_big.zarr
  python3 write_synthetic_dataset.py --config-name=synthetic_dataset \
    +output=\$HOME/seqax-offload/synthetic_big.zarr examples=65536 > /tmp/gen.log 2>&1
  pip install -q --user 'typeguard==4.1.5' 2>&1 | tail -1
  echo \"\$(hostname|rev|cut -d- -f1|rev): data=\$(du -sh ~/seqax-offload/synthetic_big.zarr 2>/dev/null|cut -f1) tg=\$(python3 -c 'from typeguard import TypeCheckerCallable;print(\"ok\")' 2>&1|tail -1)\"
" 2>&1 | grep -E "data=" | sort -u
# refresh ms-0 too so both slices run identical code
gcloud compute tpus tpu-vm scp /tmp/seqax_v4.tgz "$S0":~/seqax.tgz --zone="$ZONE" --worker=all >/dev/null 2>&1
gcloud compute tpus tpu-vm ssh "$S0" --zone="$ZONE" --worker=all --command="tar xzf ~/seqax.tgz; mkdir -p /tmp/seqax_out" >/dev/null 2>&1

echo
echo "=== stage 2: sweep (arm0 = 16 chips 1 slice, arm1 = 32 chips 2 slices) ==="
# arm0 on 16 chips, arm1 on 32: arm1 batch is twice arm0's to match tokens-per-chip.
for pair in "32 64" "64 128" "128 256" "256 512"; do
  a0=${pair% *}; a1=${pair#* }; tpc=$((64 * a0))
  echo "--- tokens/chip=$tpc   arm0 b=$a0 (16 chips)   arm1 b=$a1 (32 chips) ---"

  cleanup
  gcloud compute tpus tpu-vm ssh "$S0" --zone="$ZONE" --worker=all --command="cd ~/seqax-offload && \
    MEGASCALE_NUM_SLICES=1 python3 -m train --config-name=c4_a100x8_2b \
    +paths.model_name=v4arm0_b$a0 mesh.d=16 mesh.t=1 training.tokens.batch=$a0 $COMMON \
    > /tmp/v4arm0_b$a0.log 2>&1; echo done" >/dev/null 2>&1
  pull "$S0" "/tmp/v4arm0_b$a0.log" "v4arm0_b$a0.log"
  echo "    arm0: $(grep -oE 'Profile time: [0-9.]+' "$OUT/logs/v4arm0_b$a0.log" 2>/dev/null || echo FAILED)"

  cleanup
  for n in "$S0" "$S1"; do
    gcloud compute tpus tpu-vm ssh "$n" --zone="$ZONE" --worker=all --command="cd ~/seqax-offload && \
      python3 -m train --config-name=c4_a100x8_2b \
      +paths.model_name=v4arm1_b$a1 mesh.d=16 mesh.t=1 mesh.dcn=data:2 training.tokens.batch=$a1 \
      $COMMON > /tmp/v4arm1_b$a1.log 2>&1; echo done" >/dev/null 2>&1 &
  done
  wait
  pull "$S0" "/tmp/v4arm1_b$a1.log" "v4arm1_b$a1.log"
  echo "    arm1: $(grep -oE 'Profile time: [0-9.]+' "$OUT/logs/v4arm1_b$a1.log" 2>/dev/null || echo FAILED)"
done

cleanup
echo
echo "=== steady-state step times ==="
python3 "$HERE/tools/step_times.py" "$OUT"/logs/v4arm*.log 2>/dev/null
