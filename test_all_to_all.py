"""Tests for shardops.all_to_all.

Run on CPU with 8 simulated devices:
  SEQAX_CPU=1 XLA_FLAGS=--xla_force_host_platform_device_count=8 python -m pytest test_all_to_all.py
"""

import init_seqax  # noqa: F401  # isort: skip

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import mesh_utils
from jax.sharding import Mesh

import shardlib.shardops as shardops
import shardlib.shardtypes as shardtypes
from shardlib.shardtypes import f32, make_shardings, pytree_dataclass, typed_shard_map

shardtypes.register_with_typeguard()


def _mesh():
    return Mesh(mesh_utils.create_device_mesh([2, 4], jax.devices()), ("d", "t"))


@pytree_dataclass
class Batch:
    """A batch sharded over `t` along its length axis."""

    x: f32[b"B L/t M"]


@pytree_dataclass
class Resharded:
    """The same batch, sharded over `t` along its batch axis instead."""

    x: f32[b"B/t L M"]


def test_all_to_all_is_a_pure_reshard():
    """The global tensor is unchanged; only which axis carries the sharding changes."""
    B, L, M = 8, 16, 3
    with _mesh():
        value = jnp.arange(B * L * M, dtype=jnp.float32).reshape(B, L, M)
        value = jax.device_put(value, make_shardings(Batch).x)

        @typed_shard_map
        def move(x: f32[b"B L/t M"]) -> f32[b"B/t L M"]:
            # Per-chip shape goes from (B, L/4, M) to (B/4, L, M): same bytes, new layout.
            assert x.shape == (B, L // 4, M), x.shape
            y = shardops.all_to_all("B L/t M -> B/t L M", x)
            assert y.shape == (B // 4, L, M), y.shape
            return y

        out = move(value)
        expected_spec = make_shardings(Resharded).x.spec  # needs an active mesh

    np.testing.assert_array_equal(np.asarray(out), np.asarray(value))
    # Compared with shardlib's own notion of equivalence, since PartitionSpec('t') and
    # PartitionSpec('t', None, None) mean the same thing.
    assert shardtypes._partition_spec_equiv(out.sharding.spec, expected_spec)


def test_all_to_all_round_trips():
    """Moving the sharding there and back is the identity."""
    B, L, M = 8, 16, 3
    with _mesh():
        value = jnp.arange(B * L * M, dtype=jnp.float32).reshape(B, L, M)
        value = jax.device_put(value, make_shardings(Batch).x)

        @typed_shard_map
        def there_and_back(x: f32[b"B L/t M"]) -> f32[b"B L/t M"]:
            y = shardops.all_to_all("B L/t M -> B/t L M", x)
            return shardops.all_to_all("B/t L M -> B L/t M", y)

        out = there_and_back(value)

    np.testing.assert_array_equal(np.asarray(out), np.asarray(value))


def test_all_to_all_agrees_with_all_gather_then_slice():
    """Ground truth: all_to_all == all_gather the source axis, then keep only your slice
    of the destination axis. Same result, but all_to_all never materializes the whole
    gathered tensor -- which is the point of using it."""
    B, L, M = 8, 16, 3
    with _mesh():
        value = jnp.arange(B * L * M, dtype=jnp.float32).reshape(B, L, M)
        value = jax.device_put(value, make_shardings(Batch).x)

        @typed_shard_map
        def via_all_to_all(x: f32[b"B L/t M"]) -> f32[b"B/t L M"]:
            return shardops.all_to_all("B L/t M -> B/t L M", x)

        @typed_shard_map
        def via_all_gather(x: f32[b"B L/t M"]) -> f32[b"B/t L M"]:
            full = shardops.all_gather("B L/t M -> B L M", x)
            b_per_chip = B // 4
            start = b_per_chip * jax.lax.axis_index("t")
            return jax.lax.dynamic_slice_in_dim(full, start, b_per_chip, axis=0)

        fast = via_all_to_all(value)
        slow = via_all_gather(value)

    np.testing.assert_array_equal(np.asarray(fast), np.asarray(slow))


def test_all_to_all_moves_multiple_mesh_axes():
    """Both mesh axes can move together, matching all_gather's multi-axis support."""
    B, L, M = 8, 16, 3
    with _mesh():

        @typed_shard_map
        def move(x: f32[b"B L/t/d M"]) -> f32[b"B/t/d L M"]:
            assert x.shape == (B, L // 8, M), x.shape
            y = shardops.all_to_all("B L/t/d M -> B/t/d L M", x)
            assert y.shape == (B // 8, L, M), y.shape
            return y

        value = jnp.arange(B * L * M, dtype=jnp.float32).reshape(B, L, M)
        value = jax.device_put(value, jax.sharding.NamedSharding(_mesh(), jax.sharding.PartitionSpec(None, ("t", "d"))))
        out = move(value)

    np.testing.assert_array_equal(np.asarray(out), np.asarray(value))


def test_rejects_non_reshard_specs():
    """all_gather and psum_scatter shapes are errors: this op neither replicates nor reduces."""
    with _mesh():

        @typed_shard_map
        def replicate(x: f32[b"B L/t M"]) -> f32[b"B L M"]:
            return shardops.all_to_all("B L/t M -> B L M", x)

        with pytest.raises(ValueError, match="one axis losing sharding and one gaining"):
            replicate(jnp.zeros((8, 16, 3), dtype=jnp.float32))

        @typed_shard_map
        def mismatched(x: f32[b"B L/t M"]) -> f32[b"B/d L M"]:
            return shardops.all_to_all("B L/t M -> B/d L M", x)

        with pytest.raises(ValueError, match="drops"):
            mismatched(jnp.zeros((8, 16, 3), dtype=jnp.float32))
