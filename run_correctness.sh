#!/usr/bin/env bash
cd ~/seqax-offload
COMMON=( --config-name=c4_a100x8_270m mesh.d=4 mesh.t=1 training.tokens.batch=8
  flat_tokens.filespec=synthetic_dataset.zarr flat_tokens.streams=2
  flat_tokens.read_blocks_per_shuffle_buffer=8 flat_tokens.sequences_per_read_block=16
  training.steps=4 training.warmup_steps=1 training.steps_for_lr=4
  checkpoint_interval=1000 paths.root_working_dir=/tmp )
echo "===== OFFLOAD OFF ====="
SEQAX_CPU=1 python -m train "${COMMON[@]}" +paths.model_name=corr_off 2>&1 | grep -E "Step [0-9]|Peak HBM|True peak|error|RESOURCE"
echo "===== OFFLOAD ON ====="
SEQAX_OFFLOAD=1 SEQAX_CPU=1 python -m train "${COMMON[@]}" +paths.model_name=corr_on 2>&1 | grep -E "Step [0-9]|Peak HBM|True peak|error|RESOURCE"
