"""Validate the host<->device (CPU offload) primitives before wiring them into training.

Run on the v4-8 TPU:
    SEQAX_CPU=1 python -m test_offload

Answers three questions:
  1. Do `pinned_host` shardings work (does optimizer state actually go to CPU RAM)?
  2. Does host-resident state stay OUT of HBM (the whole point)?
  3. Do the explicit `to_host`/`to_device` transfers work at the jit level, and do they
     survive `typed_shard_map` (the open question from docs/offload_phase0.md Phase 1)?
"""

import init_seqax  # noqa: F401  # sets XLA/LIBTPU flags; must precede `import jax`

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh

import shardlib.shardops as shardops
import shardlib.shardtypes as shardtypes
from shardlib.shardtypes import f32, make_shardings, pytree_dataclass, typed_shard_map

shardtypes.register_with_typeguard()
GIB = 2**30


@pytree_dataclass
class Block:
    # A d_model x d_ff weight, sharded over the data axis `d` like seqax's real weights.
    w: f32["M/d F"]


def peak_hbm_gib():
    try:
        return jax.local_devices()[0].memory_stats()["peak_bytes_in_use"] / GIB
    except Exception:
        return float("nan")


def main():
    mesh = Mesh(mesh_utils.create_device_mesh([4, 1], jax.devices()), ("d", "t"))
    with mesh:
        M, F = 16384, 16384  # ~1.0 GiB per f32 array, global
        base = jnp.ones((M, F), jnp.float32)

        dev = make_shardings(Block)  # HBM (default)
        host = make_shardings(Block, memory_kind="pinned_host")  # CPU RAM
        print("Q1 memory_kind: device =", dev.w.memory_kind, "| host =", host.w.memory_kind)

        # --- Q2: does host-resident state stay out of HBM? ---
        w = jax.device_put(base, dev.w)
        jax.block_until_ready(w)
        hbm_after_weights = peak_hbm_gib()
        print(f"Q2 peak HBM after weights on device : {hbm_after_weights:.3f} GiB")

        mu = jax.device_put(base, host.w)
        nu = jax.device_put(base, host.w)
        jax.block_until_ready((mu, nu))
        hbm_after_optstate = peak_hbm_gib()
        print(f"   mu.memory_kind = {mu.sharding.memory_kind}")
        print(f"   peak HBM after mu,nu on host       : {hbm_after_optstate:.3f} GiB")
        print(f"   -> host optimizer state added {hbm_after_optstate - hbm_after_weights:+.3f} GiB to HBM "
              f"(want ~0.0; two 1 GiB arrays that stayed off-chip)")

        # --- Q3a: explicit round-trip at the jit level (host -> device -> update -> host) ---
        @jax.jit
        def update_jit(w, mu_host):
            mu_dev = shardops.to_device(mu_host)   # CPU -> HBM, explicit
            new_w = w - 0.1 * mu_dev
            new_mu = shardops.to_host(mu_dev)      # HBM -> CPU, explicit
            return new_w, new_mu

        new_w, new_mu = update_jit(w, mu)
        jax.block_until_ready((new_w, new_mu))
        print(f"Q3a jit round-trip OK; new_mu.memory_kind = {new_mu.sharding.memory_kind} "
              f"(want pinned_host)")

        # --- Q3b: does it survive typed_shard_map? (the docs/offload_phase0.md question) ---
        try:
            @jax.jit
            def update_sm(w, mu_host):
                @typed_shard_map
                def body(w: f32["M/d F"], mu_host: f32["M/d F"]) -> f32["M/d F"]:
                    mu_dev = shardops.to_device(mu_host)
                    return w - 0.1 * mu_dev
                return body(w, mu_host)

            r = update_sm(w, mu)
            jax.block_until_ready(r)
            print("Q3b to_device INSIDE typed_shard_map: OK  -> transfer can live inside shard_map")
        except Exception as e:
            print(f"Q3b to_device INSIDE typed_shard_map: FAILED ({type(e).__name__}) "
                  f"-> transfer must live at the OUTER jit boundary")
            print("     ", str(e).splitlines()[0][:200])

        # --- Q4: can the Adam update itself run ON the host (real ZeRO-Offload)? ---
        # For the big win, mu/nu must never touch HBM: move the gradient to host, run the
        # elementwise Adam update ON the host, and keep mu/nu resident on the host.
        try:
            from jax.experimental.compute_on import compute_on
            have_compute_on = True
        except Exception:
            have_compute_on = False

        if not have_compute_on:
            print("Q4 host-side update: `jax.experimental.compute_on` NOT importable "
                  "-> host-side update likely needs a JAX bump")
        else:
            try:
                @jax.jit
                def host_update(grad_dev, mu_host, nu_host):
                    grad_host = shardops.to_host(grad_dev)  # HBM -> CPU
                    with compute_on("device_host"):         # run the update ON the host CPU
                        new_mu = 0.9 * mu_host + 0.1 * grad_host
                        new_nu = 0.95 * nu_host + 0.05 * grad_host * grad_host
                    return new_mu, new_nu

                nm, nn = host_update(w, mu, nu)  # reuse w as a stand-in gradient
                jax.block_until_ready((nm, nn))
                hbm_after_host_update = peak_hbm_gib()
                print(f"Q4 host-side update: OK -> real ZeRO-Offload viable; "
                      f"new_mu.memory_kind = {nm.sharding.memory_kind} (want pinned_host)")
                print(f"   peak HBM after host update = {hbm_after_host_update:.3f} GiB "
                      f"(want ~unchanged; mu/nu stayed off-chip)")
            except Exception as e:
                print(f"Q4 host-side update: FAILED ({type(e).__name__}) "
                      f"-> needs JAX bump, or do the update at the outer jit boundary")
                print("     ", str(e).splitlines()[0][:200])


if __name__ == "__main__":
    main()
