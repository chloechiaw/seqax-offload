#!/usr/bin/env bash
# Rebuild a v6e slice from bare metal after a spot preemption.
#
#   ./v6e_setup.sh probe-v6e-0 [europe-west4-a]
#
# Everything here is node-local and dies with the allocation: the venv, the repo, and the
# synthetic dataset. Only the git history survives, so this script is the fastest path from
# a fresh ACTIVE node back to a runnable slice. Takes roughly 10 minutes.
#
# Every step is one of the traps documented in INHAND_SWEEP_PLAN.txt section 10. Read that
# before changing anything here.
set -euo pipefail

TPU="${1:?usage: v6e_setup.sh <node-name> [zone]}"
ZONE="${2:-europe-west4-a}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() { gcloud compute tpus tpu-vm ssh "$TPU" --zone="$ZONE" --worker=0 --command="$1"; }

echo "==> [1/5] transparent hugepages (10.5: must be set before any TPU process starts)"
run "sudo sh -c 'echo always > /sys/kernel/mm/transparent_hugepage/enabled'; cat /sys/kernel/mm/transparent_hugepage/enabled"

echo "==> [2/5] python venv + jax 0.4.35 (10.4: 24.04 image is PEP 668, needs python3.12-venv)"
run "sudo apt-get update -qq && sudo apt-get install -y -qq python3.12-venv
     rm -rf ~/jax0435-venv && python3 -m venv ~/jax0435-venv
     ~/jax0435-venv/bin/pip install -q -U pip
     ~/jax0435-venv/bin/pip install -q 'jax[tpu]==0.4.35' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"

echo "==> [3/5] device check (10.1: MEGASCALE_NUM_SLICES=1 or this hangs forever, silently)"
run "MEGASCALE_NUM_SLICES=1 ~/jax0435-venv/bin/python -c 'import jax; print(jax.__version__, jax.device_count(), jax.devices()[0].device_kind)'"

echo "==> [4/5] ship repo + deps"
tar czf /tmp/seqax_setup.tgz -C "$HERE/.." --exclude='__pycache__' --exclude='outputs' \
  --exclude='.git' --exclude='synthetic_big.zarr' "$(basename "$HERE")"
gcloud compute tpus tpu-vm scp /tmp/seqax_setup.tgz "$TPU":~/seqax.tgz --zone="$ZONE" --worker=0
run "tar xzf ~/seqax.tgz && mkdir -p /tmp/seqax_out
     ~/jax0435-venv/bin/pip install -q 'zarr<3' 'fsspec[gcs]' einops hydra-core numcodecs"

echo "==> [5/5] synthetic dataset (10.10: shipped one is ~256 sequences, far too small)"
# jaxtyping 0.2.34 is required to run the generator at all, but it pins typeguard 2.13.3
# while shardlib needs the 4.1.5 API. Generate first, then put typeguard back. The pip
# incompatibility warning in between is expected.
run "~/jax0435-venv/bin/pip install -q 'jaxtyping==0.2.34'
     cd ~/seqax-offload/tools && rm -rf ~/seqax-offload/synthetic_big.zarr
     ~/jax0435-venv/bin/python write_synthetic_dataset.py --config-name=synthetic_dataset \
       +output=\$HOME/seqax-offload/synthetic_big.zarr examples=65536 > /tmp/gen.log 2>&1
     echo \"dataset: \$(du -sh ~/seqax-offload/synthetic_big.zarr | cut -f1)\"
     ~/jax0435-venv/bin/pip install -q 'typeguard==4.1.5'
     ~/jax0435-venv/bin/python -c 'from typeguard import TypeCheckerCallable; print(\"typeguard ok\")'"

cat <<EOF

==> ready. Smoke it with:

gcloud compute tpus tpu-vm ssh $TPU --zone=$ZONE --worker=0 --command="cd ~/seqax-offload && \\
  MEGASCALE_NUM_SLICES=1 ~/jax0435-venv/bin/python -m train --config-name=c4_a100x8_2b \\
  +paths.model_name=v6e_smoke mesh.d=8 mesh.t=1 training.tokens.batch=32 \\
  training.tokens.len=1024 training.steps=50 training.warmup_steps=5 training.steps_for_lr=50 \\
  flat_tokens.filespec=synthetic_big.zarr flat_tokens.streams=2 \\
  flat_tokens.read_blocks_per_shuffle_buffer=8 flat_tokens.sequences_per_read_block=256 \\
  checkpoint_interval=100000 paths.root_working_dir=/tmp/seqax_out 2>&1 | tail -20"

Expect: 'TPU v6 lite', 'FLOPS per device: 918_000_000_000_000.0', ~261 ms/step at 22% MFU,
and 'COLLECTIVES CROSSING DCN: none' while mesh.dcn is unset.

Process cleanup goes in its OWN ssh call, never alongside what it kills (10.6).
EOF
