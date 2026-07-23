#!/usr/bin/env bash
cd ~/seqax-offload
COMMON=( mesh.d=4 mesh.t=1
  flat_tokens.filespec=synthetic_dataset.zarr flat_tokens.streams=2
  flat_tokens.read_blocks_per_shuffle_buffer=8 flat_tokens.sequences_per_read_block=16
  training.steps=4 training.warmup_steps=1 training.steps_for_lr=4
  checkpoint_interval=1000 paths.root_working_dir=/tmp )
for SZ in 84m 270m 540m 1b 2b; do
  echo "===== $SZ ====="
  SEQAX_CPU=1 python -m train --config-name=c4_a100x8_${SZ} \
    +paths.model_name=v4_${SZ} "${COMMON[@]}" 2>&1 | tee ~/phase0_v4_${SZ}.log
done
