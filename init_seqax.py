import os
import sys


def set_variables():
    _gpu_flags = ""
    if os.environ.get("SEQAX_CPU") != "1":
        _gpu_flags = "--xla_gpu_enable_async_collectives=true --xla_gpu_enable_latency_hiding_scheduler=true "
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " " + _gpu_flags
    os.environ.update(
        {
            "NCCL_LL128_BUFFSIZE": "-2",
            "NCCL_LL_BUFFSIZE": "-2",
            "NCCL_PROTO": "SIMPLE,LL,LL128",
        }
    )
    os.environ["LIBTPU_INIT_ARGS"] = (
        "--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
    )


# Set environment variables
assert "jax" not in sys.modules, (
    "JAX requires XLA_FLAGS to be set before importing it (https://docs.jax.dev/en/latest/xla_flags.html)"
)
set_variables()
