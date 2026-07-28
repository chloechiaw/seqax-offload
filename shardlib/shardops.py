import jax
import jax.numpy as jnp
from jax import lax

import shardlib.shardtypes as shardtypes


def all_gather(spec: str, x):
    """String-specified all-gather operation.

    For example:
      all_gather('A/x/y B/z C/w -> A B C/w', x)
    """
    before, after = spec.split("->")
    before = shardtypes.ShapeSpec.parse(before)
    after = shardtypes.ShapeSpec.parse(after)
    shardtypes.check(x.dtype, before, x)
    for i, (before_dim, after_dim) in enumerate(zip(before.dims, after.dims)):
        # Check that after_dim.sharding is a prefix of before_dim.sharding
        after_n = len(after_dim.sharding)
        if before_dim.shape != after_dim.shape or before_dim.sharding[:after_n] != after_dim.sharding:
            raise ValueError(f"Cannot all-gather {before_dim} into {after_dim}")
        if len(before_dim.sharding) == after_n:
            continue
        x = lax.all_gather(x, tuple(before_dim.sharding[after_n:]), axis=i, tiled=True)
    shardtypes.check(x.dtype, after, x)
    return x


def psum_scatter(spec: str, x):
    """String-specified reduce-scatter operation.

    For example:
      psum_scatter('A B C/w -> A/x/y B/z C/w', x)
    """
    before, after = spec.split("->")
    before = shardtypes.ShapeSpec.parse(before)
    after = shardtypes.ShapeSpec.parse(after)
    shardtypes.check(x.dtype, before, x)
    for i, (before_dim, after_dim) in enumerate(zip(before.dims, after.dims)):
        # Check that before_dim.sharding is a prefix of after_dim.sharding
        before_n = len(before_dim.sharding)
        if before_dim.shape != after_dim.shape or after_dim.sharding[:before_n] != before_dim.sharding:
            raise ValueError(f"Cannot reduce-scatter {before_dim} into {after_dim}")
        if len(after_dim.sharding) == before_n:
            continue
        x = lax.psum_scatter(x, tuple(after_dim.sharding[before_n:]), scatter_dimension=i, tiled=True)
    shardtypes.check(x.dtype, after, x)
    return x


def all_to_all(spec: str, x):
    """String-specified all-to-all operation, which moves a sharding between two axes.

    Unlike all_gather and psum_scatter, this neither replicates nor reduces: the tensor
    stays fully sharded over the same mesh axes, and the same total bytes are resident
    before and after. Only *which* tensor axis carries the sharding changes.

    For example:
      all_to_all('B L/t M -> B/t L M', x)

    Here `x` is sharded over `t` along its length axis on input, and over `t` along its
    batch axis on output. Each chip sends every other chip the slice of `B` that chip
    will own, and receives the slices of `L` it was missing.

    Cost: with `n` chips on the moved axis, each chip sends and receives `(n-1)/n` of its
    local tensor, versus `(n-1)/n` sent-and-received-and-*retained* for an all_gather of
    the same tensor. So the wire traffic is comparable to an all_gather, but the memory
    afterwards is `1/n` of it -- which is the whole reason to reach for this operation.

    Exactly one axis must lose sharding and exactly one must gain it, and they must move
    the same mesh axes.
    """
    before, after = spec.split("->")
    before = shardtypes.ShapeSpec.parse(before)
    after = shardtypes.ShapeSpec.parse(after)
    shardtypes.check(x.dtype, before, x)
    if len(before.dims) != len(after.dims):
        raise ValueError(f"Cannot all-to-all {before} into {after}: different ranks")

    src = None  # axis losing sharding, so it gets concatenated back together
    dst = None  # axis gaining sharding, so it gets split up
    for i, (before_dim, after_dim) in enumerate(zip(before.dims, after.dims)):
        if before_dim.shape != after_dim.shape:
            raise ValueError(f"Cannot all-to-all {before_dim} into {after_dim}: shapes differ")
        if before_dim.sharding == after_dim.sharding:
            continue
        after_n = len(after_dim.sharding)
        before_n = len(before_dim.sharding)
        if before_dim.sharding[:after_n] == after_dim.sharding:
            if src is not None:
                raise ValueError(f"Cannot all-to-all {before} into {after}: more than one axis loses sharding")
            src = i
        elif after_dim.sharding[:before_n] == before_dim.sharding:
            if dst is not None:
                raise ValueError(f"Cannot all-to-all {before} into {after}: more than one axis gains sharding")
            dst = i
        else:
            raise ValueError(f"Cannot all-to-all {before_dim} into {after_dim}: shardings are not nested")

    if src is None or dst is None:
        raise ValueError(
            f"Cannot all-to-all {before} into {after}: need exactly one axis losing sharding and one gaining it. "
            f"To replicate a sharding use all_gather; to reduce one use psum_scatter."
        )

    moved = tuple(before.dims[src].sharding[len(after.dims[src].sharding) :])
    gained = tuple(after.dims[dst].sharding[len(before.dims[dst].sharding) :])
    if moved != gained:
        raise ValueError(f"Cannot all-to-all {before} into {after}: axis {src} drops {moved} but axis {dst} adds {gained}")

    # `split_axis` is the axis we cut into per-destination-chip pieces (the one gaining the
    # sharding); `concat_axis` is the axis we glue the received pieces onto (the one losing it).
    x = lax.all_to_all(x, moved, split_axis=dst, concat_axis=src, tiled=True)
    shardtypes.check(x.dtype, after, x)
    return x


def einsum_unreduced(spec: str, x, y, **kwargs):
    """Ordinary chip-local einsum, but with sharding-aware typechecking.

    Note that this function does not do any chip-to-chip communication. If the inputs are
    sharded over the contraction dimensions, the caller is responsible for reducing the result
    over those dimensions. For example:

      c = einsum_unreduced('A/x B/y, B/y C/z -> A/x/z', a, b)
      # c still needs to be reduced over the y axis.
      d = psum_scatter('A/x/z -> A/x/z/y', c)
      # Now the post-einsum reduction is complete.
    """
    tmp, result = spec.split("->")
    lhs, rhs = tmp.split(",")
    lhs = shardtypes.ShapeSpec.parse(lhs)
    rhs = shardtypes.ShapeSpec.parse(rhs)
    result = shardtypes.ShapeSpec.parse(result)
    shardtypes.check(x.dtype, lhs, x)
    shardtypes.check(y.dtype, rhs, y)
    # Convert to jax einsum syntax, with single-letter variables.
    jaxspec = ""

    vars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    var_i = 0
    dim_table = {}

    def map_var(dim):
        if dim in dim_table:
            return dim_table[dim]
        nonlocal var_i
        if var_i >= len(vars):
            raise ValueError("Too many dimensions in einsum, we ran out of variables")
        var = vars[var_i]
        var_i += 1
        dim_table[dim] = var
        return var

    for dim in lhs.dims:
        jaxspec += map_var(dim)
    jaxspec += ","
    for dim in rhs.dims:
        jaxspec += map_var(dim)
    jaxspec += "->"
    for dim in result.dims:
        jaxspec += map_var(dim)
    r = jnp.einsum(jaxspec, x, y, **kwargs)
    shardtypes.check(r.dtype, result, r)
    return r


def index_unreduced(spec: str, table, indices):
    """String-specified sharded table lookup operation.

    For example:
      index_unreduced(table, indices, 'A [B/x/y] C/z, D/w A -> C/z A D/w')

    In this example, the integers in `indices` are used as lookup addresses into the
    `B` dimension of `table`, and all other dimensions (`A`, `C`, `D`) are vmapped over.

    This operation does not do any chip-to-chip communication, even though the table
    may be sharded. If the axis inside square brackets is sharded, corresponding to
    different table indices on different shards, a table lookup will be performed on each
    shard, but only one shard will return a nonzero result: the other shards, where the
    index is out of bounds, will return zero. The caller is required to reduce the output
    over the axes specified by the square brackets: in the above example, the caller must
    reduce over `x` and `y` axes.
    """
    tmp, result = spec.split("->")
    lhs, rhs = tmp.split(",")
    lhs_dims = lhs.split()
    index_axis = None
    for i, dim in enumerate(lhs_dims):
        if dim.startswith("["):
            index_axis = i
            if not dim.endswith("]"):
                raise ValueError(f"Expected closing bracket in {dim}")
            lhs_dims[i] = dim[1:-1]
            break
    if index_axis is None:
        raise ValueError(f"Expected an index axis in {lhs}")

    lhs_dims = [shardtypes.DimSpec.parse(dim) for dim in lhs_dims]
    lhs_spec = shardtypes.ShapeSpec(lhs_dims)
    rhs_spec = shardtypes.ShapeSpec.parse(rhs)
    result_spec = shardtypes.ShapeSpec.parse(result)
    shardtypes.check(table.dtype, lhs_spec, table)
    shardtypes.check(indices.dtype, rhs_spec, indices)

    # Do the base operation on scalars, then do a sequence of vmap operations to bring it up
    # to the desired shape.
    def base_op(table, index):
        len_per_chip = table.shape[0]
        lower_bound = len_per_chip * lax.axis_index(lhs_dims[index_axis].sharding)
        upper_bound = lower_bound + len_per_chip
        in_bounds = (lower_bound <= index) & (index < upper_bound)
        return jnp.where(in_bounds, table[jnp.where(in_bounds, index - lower_bound, 0)], 0)

    op = base_op

    lhs_dims_handled = [False] * len(lhs_dims)
    lhs_dims_handled[index_axis] = True
    rhs_dims_handled = [False] * len(rhs_spec.dims)
    for dim in reversed(result_spec.dims):
        try:
            lhs_index = lhs_dims.index(dim)
            lhs_vmap_axis = sum(lhs_dims_handled[:lhs_index])
            assert not lhs_dims_handled[lhs_index]
            lhs_dims_handled[lhs_index] = True
        except ValueError:
            lhs_index = None
            lhs_vmap_axis = None

        try:
            rhs_index = rhs_spec.dims.index(dim)
            rhs_vmap_axis = sum(rhs_dims_handled[:rhs_index])
            assert not rhs_dims_handled[rhs_index]
            rhs_dims_handled[rhs_index] = True
        except ValueError:
            rhs_index = None
            rhs_vmap_axis = None

        op = jax.vmap(op, in_axes=(lhs_vmap_axis, rhs_vmap_axis), out_axes=0)

    assert all(lhs_dims_handled)
    assert all(rhs_dims_handled)

    result = op(table, indices)
    shardtypes.check(result.dtype, result_spec, result)
    return result


def axis_size(name: str) -> int:
    """Return the size of the axis with the given name."""
    return jax.lax.psum(1, name)


# ---- Host <-> device memory placement --------------------------------------------------
# seqax makes cross-*chip* movement explicit (all_gather / psum_scatter). These make
# cross-*memory* movement explicit in the same spirit: a visible operation in the code that
# moves a tensor between TPU HBM ("device") and pinned host CPU RAM ("pinned_host"), rather
# than a hidden compiler decision. Used to park optimizer state in host RAM between steps and
# stream it back to HBM only for the weight update.


try:
    from jax._src.sharding_impls import TransferToMemoryKind as _TransferToMemoryKind
except Exception:
    _TransferToMemoryKind = None


def _rememory(x, memory_kind: str):
    """Return `x` placed in the given memory space, preserving its (chip) sharding.

    Works inside jit: there `x` is a tracer with no concrete `.sharding`, so we use
    `TransferToMemoryKind`, which changes only the memory space and keeps the sharding.
    """
    if _TransferToMemoryKind is not None:
        return jax.device_put(x, _TransferToMemoryKind(memory_kind))
    # Fallback (older JAX, or a concrete array outside jit): rebuild the NamedSharding.
    s = x.sharding
    return jax.device_put(x, jax.sharding.NamedSharding(s.mesh, s.spec, memory_kind=memory_kind))


def to_host(x):
    """Explicitly move `x` from TPU HBM to pinned host (CPU) RAM.

    The host-memory analogue of `all_gather`: an explicit data movement across the
    host<->device boundary instead of across chips. The chip-sharding is unchanged.
    """
    return _rememory(x, "pinned_host")


def to_device(x):
    """Explicitly move `x` from pinned host (CPU) RAM back to TPU HBM."""
    return _rememory(x, "device")
