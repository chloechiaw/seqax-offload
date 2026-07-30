#!/usr/bin/env bash
# Provision every node of a v4 carve: repo, deps, and the synthetic dataset on every worker.
#
#   ./v4_setup.sh seqax-4x16-0 seqax-4x16-1 seqax-4x16-2 seqax-4x16-3
#
# Runs all nodes in parallel, which matters: a 4-slice carve is 4 nodes and the dataset has to
# be generated on each worker separately (every process reads it locally). Serially this is
# ~40 minutes; in parallel it is closer to 10.
#
# v4 images are Ubuntu 22.04, so no PEP 668 and no venv needed -- pip --user works. But a
# FRESHLY provisioned node has no jax at all; the older seqax-ms nodes only had it because it
# had been installed by hand earlier. Install it explicitly or every run dies with
# ModuleNotFoundError after the dataset has already been generated.
set -uo pipefail

ZONE="${ZONE:-us-central2-b}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ $# -ge 1 ] || { echo "usage: v4_setup.sh <node> [node...]"; exit 1; }

tar czf /tmp/seqax_v4carve.tgz -C "$HERE/.." --exclude='__pycache__' --exclude='outputs' \
  --exclude='.git' --exclude='synthetic_big.zarr' --exclude='results' "$(basename "$HERE")"

for node in "$@"; do
(
  gcloud compute tpus tpu-vm scp /tmp/seqax_v4carve.tgz "$node":~/seqax.tgz \
    --zone="$ZONE" --worker=all >/dev/null 2>&1
  gcloud compute tpus tpu-vm ssh "$node" --zone="$ZONE" --worker=all --command="
    tar xzf ~/seqax.tgz && mkdir -p /tmp/seqax_out
    pip install -q --user 'jax[tpu]==0.4.35' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 2>&1 | tail -1
    pip install -q --user 'zarr<3' 'fsspec[gcs]' einops hydra-core numcodecs 'jaxtyping==0.2.34' 2>&1 | tail -1
    cd ~/seqax-offload/tools && rm -rf ~/seqax-offload/synthetic_big.zarr
    python3 write_synthetic_dataset.py --config-name=synthetic_dataset \
      +output=\$HOME/seqax-offload/synthetic_big.zarr examples=65536 > /tmp/gen.log 2>&1
    pip install -q --user 'typeguard==4.1.5' 2>&1 | tail -1
  " >/dev/null 2>&1
  ok=$(gcloud compute tpus tpu-vm ssh "$node" --zone="$ZONE" --worker=all --command="
    echo \"\$(du -sh ~/seqax-offload/synthetic_big.zarr 2>/dev/null|cut -f1) \$(python3 -c 'from typeguard import TypeCheckerCallable;print(\"tg-ok\")' 2>&1|tail -1)\"
  " 2>/dev/null | grep -c "tg-ok")
  echo "  $node: $ok workers ready"
) &
done
wait

echo
echo "device check across all nodes (each slice should report its own chip count):"
for node in "$@"; do
  gcloud compute tpus tpu-vm ssh "$node" --zone="$ZONE" --worker=0 \
    --command="MEGASCALE_NUM_SLICES=1 python3 -c 'import jax; print(\"$node\", jax.device_count(), \"chips\")'" 2>/dev/null | grep chips
done
