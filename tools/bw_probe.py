"""Measure achieved all-reduce bandwidth over a named mesh axis.

  python3 tools/bw_probe.py dcn    # over `data`: between slices, the slow link
  python3 tools/bw_probe.py ici    # over `d`:    within a slice, the fast link

No model, no framework, no confounds -- one psum of a known size, timed. The ICI number is
the denominator the whole "hierarchy must match the network" argument rests on: it is only
worth putting weight traffic on ICI and gradient traffic on DCN if the two differ by a lot,
and this is what says by how much.

Run on all slices at once for dcn; a single slice suffices for ici.
"""

import sys
import time

import jax
import jax.experimental.shard_map as sm
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec

which = sys.argv[1] if len(sys.argv) > 1 else "dcn"
n_slices = int(sys.argv[2]) if len(sys.argv) > 2 else (2 if which == "dcn" else 1)

per_slice = jax.device_count() // n_slices
ici_shape, dcn_shape = (1, per_slice, 1), (n_slices, 1, 1)
if n_slices > 1:
    devs = mesh_utils.create_hybrid_device_mesh(ici_shape, dcn_shape, jax.devices())
else:
    devs = mesh_utils.create_device_mesh(list(ici_shape), jax.devices())
mesh = Mesh(devs, ("data", "d", "t"))

axis = "data" if which == "dcn" else "d"
participants = n_slices if which == "dcn" else per_slice
first = jax.process_index() == 0

if first:
    print(f"link={which} axis={axis} participants={participants} "
          f"devices={jax.device_count()} slices={n_slices} per_slice={per_slice}", flush=True)
    if participants < 2:
        print("  (only one participant on this axis -- nothing to measure)", flush=True)

with mesh:
    for mib in (16, 64, 256, 1024):
        n = mib * 1024 * 1024 // 4
        x = jnp.ones((n,), jnp.float32)
        f = jax.jit(sm.shard_map(lambda v: jax.lax.psum(v, axis), mesh=mesh,
                                 in_specs=PartitionSpec(None), out_specs=PartitionSpec(None),
                                 check_rep=False))
        jax.block_until_ready(f(x))
        reps = 5
        t0 = time.perf_counter()
        for _ in range(reps):
            r = f(x)
        jax.block_until_ready(r)
        dt = (time.perf_counter() - t0) / reps
        moved = 2 * (participants - 1) / participants * mib * 1024**2
        if first:
            print(f"  buf={mib:5d} MiB  time={dt * 1e3:8.1f} ms  moved={moved / 2**20:7.1f} MiB  "
                  f"achieved={moved / dt / 1e9:6.2f} GB/s per chip", flush=True)
