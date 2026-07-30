"""Type annotations for JAX arrays with sharding information.

# Shape checking

Example:

```
import jax
shardtypes.register_with_typeguard()
from shardlib.shardtypes import f32
from typeguard import typechecked

@typechecked
def center_channels(x: f32[b'batch/d channels']) -> f32[b'batch/d channels']:
  return x - jax.numpy.mean(x, axis=-1, keepdims=True)
```

The type syntax is `<dtype>[<shape string>]`, where `dtype` is imported from `shardlib.shardtypes`,
and `<shape string>` is a space-separated list of dimensions. Each dimension consists of a dimension
name (e.g. `batch`), optionally followed by slashes and sharding axis names, e.g. `batch/d` indicates
that the `batch` tensor dimension is sharded over the `d` device axis. Sharding over multiple axes
is indicated by multiple axis names, e.g. `batch/d/e`.

The shape string may be either a string ('foo') or a bytes object (b'foo'). Strings have special
meaning in Python type annotations (they are used for forward references, and are eval'ed by typeguard),
so the bytes object b'foo' is a workaround to prevent this eval'ing.

Shape checking proceeds by maintaining a table of the sizes of all dimension names in a context
variable, known as the shape checking scope. The first time a dimension name is encountered,
its size is recorded in the current scope. Subsequent uses of the same dimension name must have
the same size. Device axes (e.g. `/d`) are looked up in the currently configured JAX device mesh,
to determine the size of the axis.

For calls into functions or libraries, it can be useful to clear the shape checking scope, so caller
and callee can use the same variable name to mean different things. This can be done with the `@scope`
function decorator or the `with Scope():` context manager.

# Using type annotations

In addition to driving shape checking, type annotations can be used to drive sharding in JAX functions.
See for example `typed_shard_map`, which is a simplification of JAX's `shard_map` by taking advantage
of sharding in type signatures.
"""

import dataclasses
import inspect
import typing
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass, make_dataclass
from types import GenericAlias
from typing import Union, get_args, get_origin

import jax
import jax.numpy as jnp
from typeguard import TypeCheckerCallable, TypeCheckError, check_type_internal, checker_lookup_functions, typechecked

#### State
# ContextVar(dict[str, int])
_VARS = ContextVar("shardtypes._VARS", default={})


class Scope:
    """Context manager that clears the shape checking scope."""

    def __enter__(self):
        self.token = _VARS.set({})

    def __exit__(self, type, value, traceback):
        _VARS.reset(self.token)


def scope(f):
    """Function decorator that clears the shape checking scope."""

    def wrapper(*args, **kwargs):
        with Scope():
            return f(*args, **kwargs)

    return wrapper


def check_size(name: str, size: int):
    """Checks that a dimension has the expected size."""
    try:
        value = int(name)
        if value != size:
            raise TypeCheckError(f"explicit dimension {value}: actually was {size}")
    except ValueError:
        v = _VARS.get()
        if name in v:
            if v[name] != size:
                raise TypeCheckError(f"dimension {name}: expected {v[name]}, got {size}")
        else:
            v[name] = size


#### Shape specs
@dataclass(frozen=True)
class DimSpec:
    """Parsed result of a dimension in a shape string."""

    shape: str
    sharding: Sequence[str]

    @staticmethod
    def parse(spec: str) -> "DimSpec":
        pieces = spec.split("/")
        shape = pieces[0]
        sharding = tuple(pieces[1:])
        return DimSpec(shape, sharding)

    def __str__(self):
        return "/".join([self.shape] + list(self.sharding))


@dataclass
class ShapeSpec:
    """Parsed result of a shape string."""

    dims: Sequence[DimSpec]

    @staticmethod
    def parse(spec: Union[bytes, str]) -> "ShapeSpec":
        if isinstance(spec, bytes):
            spec = spec.decode("utf-8")
        if not isinstance(spec, str):
            print(spec)
            raise ValueError("Expected a string")
        dims = spec.split()  # Split on spaces, trimming excess space
        result = []
        for dim in dims:
            result.append(DimSpec.parse(dim))
        return ShapeSpec(result)

    def partition_spec(self) -> jax.sharding.PartitionSpec:
        result = []
        for dim_spec in self.dims:
            if len(dim_spec.sharding) == 0:
                result.append(None)
            elif len(dim_spec.sharding) == 1:
                result.append(dim_spec.sharding[0])
            else:
                result.append(tuple(dim_spec.sharding))
        return jax.sharding.PartitionSpec(*result)

    def __str__(self):
        return " ".join(str(dim) for dim in self.dims)


#### Shape checking
def _partition_spec_equiv(lhs: jax.sharding.PartitionSpec, rhs: jax.sharding.PartitionSpec) -> bool:
    if len(lhs) < len(rhs):
        lhs, rhs = rhs, lhs
    if any(l is not None for l in lhs[len(rhs) :]):
        return False
    return lhs[: len(rhs)] == rhs[:]


def check(dtype, shape_spec: ShapeSpec, value):
    """Checks that a value has the expected dtype and shape."""
    if not isinstance(value, jax.Array):
        raise TypeCheckError("is not a jax.Array")
    if value.dtype != dtype:
        raise TypeCheckError(f"is {value.dtype}, but expected {dtype}")
    shape = value.shape
    if len(shape) != len(shape_spec.dims):
        raise TypeCheckError(f"has shape {shape}, but expected shape {str(shape_spec)}")
    mesh = None

    axis_env = jax._src.core.thread_local_state.trace_state.axis_env
    if axis_env:
        # We're in a shard_map/pmap/xmap context. Multiply sizes by sharding, then check sizes.
        # We don't actually check the sharding, because that information is lost inside a
        # shard_map/pmap/xmap context, but we do check the unsharded sizes are correct.
        mesh = {axis.name: axis.size for axis in axis_env}
        for orig_dim, dim_spec in zip(shape, shape_spec.dims):
            dim = orig_dim
            for axis in dim_spec.sharding:
                if axis not in mesh:
                    raise TypeCheckError(f"has unknown mesh axis {axis}")
                axis_size = mesh[axis]
                dim *= axis_size
            check_size(dim_spec.shape, dim)
    else:
        # Check sizes
        for dim, dim_spec in zip(shape, shape_spec.dims):
            check_size(dim_spec.shape, dim)

        # Check sharding
        expected_spec = shape_spec.partition_spec()

        def cb(actual):
            if isinstance(actual, jax.sharding.SingleDeviceSharding):
                if any(dim_spec.sharding for dim_spec in shape_spec.dims):
                    raise TypeCheckError(f"is fully replicated, but expected {expected_spec} is not")
            elif not isinstance(actual, jax.sharding.NamedSharding):
                if isinstance(actual, jax.sharding.Sharding):
                    raise TypeCheckError(
                        "is SPMD-sharded but no axis names are available. Use `with Mesh(...):` to provide axis names for type checking."
                    )
                else:
                    raise TypeCheckError(f": unexpected object when checking sharding: {actual}")
            elif not _partition_spec_equiv(actual.spec, expected_spec):
                # TODO: when an axis size is None, recovering the NamedSharding from the PositionalSharding
                # is ambiguous, and JAX often takes a different approach than the user does.
                #
                # We could fix this with a more precise _partition_spec_equiv, but for now we'll just ignore it.
                # raise TypeCheckError(f'has sharding spec {actual.spec}, but expected {expected_spec} from {str(shape_spec)}')
                pass

        # Use tracing as a proxy for whether we're in a jit context
        is_tracing = jax._src.core.thread_local_state.trace_state.trace_stack
        if is_tracing:
            jax.debug.inspect_array_sharding(value, callback=cb)
        else:
            cb(value.sharding)


#### Typeguard
def register_with_typeguard():
    """Registers the shardtypes module with typeguard. Call this at the beginning of your program."""

    def check_array(value, origin, args, memo):
        if len(args) != 1 or (type(args[0]) is not str and type(args[0]) is not bytes):
            raise TypeCheckError(
                f"has bad type signature; expected {origin.__name__}[<shape string>], got {origin.__name__}{args}"
            )
        check(origin.dtype, ShapeSpec.parse(args[0]), value)

    def check_pytree_dataclass(value, origin, args, memo):
        if not isinstance(value, origin):
            raise TypeCheckError(f"is not an instance of {origin}")
        for field in dataclasses.fields(origin):
            check_type_internal(getattr(value, field.name), field.type, memo)

    def lookup(origin, args, extras) -> TypeCheckerCallable | None:
        if isinstance(origin, type) and issubclass(origin, number):
            return check_array
        if origin in _PYTREE_DATACLASSES:
            return check_pytree_dataclass
        return None

    checker_lookup_functions.append(lookup)


#### Array types
class number:
    def __class_getitem__(cls, x):
        if isinstance(x, str):
            x = x.encode("utf-8")
        return GenericAlias(cls, x)


class bool_(number):
    dtype = jnp.bool_
    pass


class bf16(number):
    dtype = jnp.bfloat16
    pass


class f32(number):
    dtype = jnp.float32
    pass


class i32(number):
    dtype = jnp.int32
    pass


class u32(number):
    dtype = jnp.uint32
    pass


class i8(number):
    dtype = jnp.int8
    pass


class u8(number):
    dtype = jnp.uint8
    pass


_PYTREE_DATACLASSES = set()


def pytree_dataclass(cls):
    """Decorator that declares a dataclass that JAX recognizes as a PyTree."""
    cls = dataclass(cls)

    def flatten_with_keys(value):
        return [(k.name, getattr(value, k.name)) for k in dataclasses.fields(cls)], ()

    def unflatten(_aux, fields):
        return cls(*fields)

    jax.tree_util.register_pytree_with_keys(cls, flatten_with_keys, unflatten)
    _PYTREE_DATACLASSES.add(cls)
    return cls


class Array:
    """If `cls` is an array type or a `pytree_dataclass` of array types,
    `Array[axes, cls]` will extend `cls` with leading axes `axes`.
    For example, `Array['layers', f32['batch d_model']] returns f32['layers batch d_model`]`.
    """

    def __class_getitem__(cls, x):
        axes, input_cls = x
        if isinstance(axes, str):
            axes = axes.encode("utf-8")
        elif isinstance(axes, bytes):
            pass
        else:
            raise ValueError(f"input axes to {cls} must be Union[bytes, str]")

        if dataclasses.is_dataclass(input_cls):
            extended_fields = []
            for fld in dataclasses.fields(input_cls):
                extended_type = Array[axes, fld.type]
                extended_fields.append((fld.name, extended_type))

            extended_cls = make_dataclass(input_cls.__name__, extended_fields, bases=(input_cls,))
            pytree_dataclass(extended_cls)
            return extended_cls
        else:
            number_type, shape = get_origin(input_cls), get_args(input_cls)
            extended_shape = (axes + b" " + shape[0],)
            return GenericAlias(number_type, extended_shape)


def make_partition_specs(cls):
    """Instantiates a pytree dataclass with a PartitionSpec at array type."""
    # Check for a tuple type:
    origin = typing.get_origin(cls)
    args = typing.get_args(cls)
    if origin is tuple:
        return tuple(make_partition_specs(arg) for arg in args)
    elif origin is not None and issubclass(origin, number):
        if len(args) != 1 or (type(args[0]) is not str and type(args[0]) is not bytes):
            raise ValueError(f"Type annotation {cls} should be <dtype>[<shape string>], got {cls}")
        spec = ShapeSpec.parse(args[0])
        return spec.partition_spec()
    elif dataclasses.is_dataclass(cls):
        values = []
        for field in dataclasses.fields(cls):
            values.append(make_partition_specs(field.type))
        return cls(*values)

    raise ValueError(f"Unsupported type {cls} is not a array, dataclass, or tuple type")


def make_shardings(cls, memory_kind=None):
    """Instantiates a pytree dataclass with NamedSharding at array type.

    If `memory_kind` is given (e.g. "pinned_host"), the arrays are placed in that memory space
    instead of the default TPU HBM ("device"). This expresses *where a tensor lives* (host CPU vs
    TPU) in the same type-driven style that `/d` and `/t` express *how it is split across chips*.
    """
    mesh = jax._src.mesh.thread_resources.env.physical_mesh

    def _sharding(spec):
        if memory_kind is None:
            return jax.sharding.NamedSharding(mesh, spec)
        return jax.sharding.NamedSharding(mesh, spec, memory_kind=memory_kind)

    return jax.tree_util.tree_map(_sharding, make_partition_specs(cls))


def typed_shard_map(f, **kwargs):
    """jax.shard_map, but which does not require specifying in_specs and out_specs.

    Instead, the function signature is used to infer the partitioning of the inputs and outputs.

    For example:
      @typed_shard_map
      def f(x: f32[b'batch/d len'], y: f32[b'e/d f/t']) -> f32[b'batch/d f/t']:
        ...

    """
    sig = inspect.signature(f)

    def wrapped(*args):
        mesh = jax._src.mesh.thread_resources.env.physical_mesh
        in_specs = tuple(make_partition_specs(param.annotation) for param in sig.parameters.values())
        out_specs = make_partition_specs(sig.return_annotation)
        return jax.experimental.shard_map.shard_map(
            typechecked(f), in_specs=in_specs, out_specs=out_specs, mesh=mesh, **kwargs
        )(*args)

    return wrapped


#### Which mesh axes are carried by the slow network
#
# seqax's thesis is that all inter-chip communication should be explicit in the source. On a
# single slice that is enough, because every axis costs the same. Across slices it is not:
# an axis carried by DCN is roughly an order of magnitude slower per byte than one carried by
# ICI, so `A/d -> A` and `A/data -> A` are the same notation for operations whose costs differ
# by that factor.
#
# So the mesh declares which of its axes are DCN, and the collectives in shardops report what
# they push across those axes. The result is a trace-time answer to "which of my collectives
# touch the slow network, and how many bytes", which is otherwise only visible in a profile
# after a run has already cost you the time.

_DCN_AXES: frozenset = frozenset()
_REPLICA_AXES: frozenset = frozenset()
_AXIS_SIZES: dict = {}
_DCN_LOG: dict = {}
_DCN_STRICT: frozenset = frozenset()


@dataclass(frozen=True)
class DcnCollective:
    """One collective, at one source location, that crosses DCN."""

    site: str  # "file.py:123"
    op: str  # "all_gather" | "psum_scatter" | "all_reduce" | "all_to_all"
    axes: tuple  # the DCN axis names it travels over
    bytes_per_chip: int  # bytes this chip pushes across DCN, per call


def declare_mesh(mesh, dcn_axes: Sequence[str] = (), replica_axes: Sequence[str] = ()):
    """Record the mesh's axis sizes and the role of each axis. Call once, at mesh construction.

    Two independent properties, easy to conflate:

      dcn_axes      carried by the slow network. Determines what gets ACCOUNTED as DCN
                    traffic. Empty means a single slice, which makes the accounting inert.

      replica_axes  weights are replicated over these rather than sharded. Determines what
                    counts as "fully sharded" for the optimizer's assertion.

    They usually name the same axis -- `data` is both replicated-over and DCN-carried in the
    two-slice config -- but not always. On a single slice `data` still exists with size 1 and
    is still a replica axis, while nothing is DCN.
    """
    global _DCN_AXES, _REPLICA_AXES, _AXIS_SIZES
    _AXIS_SIZES = {name: size for name, size in zip(mesh.axis_names, mesh.devices.shape)}
    for label, names in (("dcn_axes", dcn_axes), ("replica_axes", replica_axes)):
        unknown = [a for a in names if a not in _AXIS_SIZES]
        if unknown:
            raise ValueError(f"{label} {unknown} are not axes of the mesh {tuple(_AXIS_SIZES)}")
    _DCN_AXES = frozenset(dcn_axes)
    _REPLICA_AXES = frozenset(replica_axes)
    reset_dcn_log()


def dcn_axes() -> frozenset:
    return _DCN_AXES


def axis_sizes() -> dict:
    return dict(_AXIS_SIZES)


def all_axes() -> tuple:
    """Every mesh axis, ICI and DCN alike.

    Use this wherever a reduction is meant to be global -- loss, token counts, gradient norm.
    Hardcoding ("d", "t") was correct while those were the only axes; once a DCN axis exists,
    the same code silently reduces over one slice only and reports per-slice numbers as if
    they were global.
    """
    return tuple(_AXIS_SIZES)


def set_dcn_allowlist(sites: Sequence[str] = ()):
    """Only these source sites may cross DCN; any other DCN collective raises at trace time.

    A misconfigured mesh -- weight gathering landing on DCN instead of gradient reduction --
    costs an order of magnitude in step time and otherwise shows up only as "training is slow".
    This turns it into a startup failure. Sites are matched as suffixes, e.g. "train.py:326".
    """
    global _DCN_STRICT
    _DCN_STRICT = frozenset(sites)


def reset_dcn_log():
    _DCN_LOG.clear()


def _caller_site() -> str:
    """Nearest stack frame outside shardlib, as file:line."""
    import os as _os
    import traceback as _tb

    for frame in reversed(_tb.extract_stack()[:-1]):
        if _os.sep + "shardlib" + _os.sep not in frame.filename:
            return f"{_os.path.basename(frame.filename)}:{frame.lineno}"
    return "<unknown>"


# Per-chip bytes moved by a ring collective over `n` participants, as a multiple of the
# per-chip tensor size going in. These are the standard ring costs, and they are the same
# formulas the experiment plan does by hand.
_RING_COST = {
    "all_gather": lambda n: n - 1,  # receives the other n-1 shards
    "psum_scatter": lambda n: (n - 1) / n,  # reduce-scatter
    "all_to_all": lambda n: (n - 1) / n,
    "all_reduce": lambda n: 2 * (n - 1) / n,  # reduce-scatter + all-gather
}


def record_collective(op: str, axes: Sequence[str], x):
    """Note that `op` travels over `axes`. No-op unless some axis is DCN.

    Called from shardops for every collective. Runs at trace time, so it costs nothing at
    runtime and reports before a single step has executed.
    """
    record_collective_bytes(op, axes, x.size * jnp.dtype(x.dtype).itemsize)


def record_collective_bytes(op: str, axes: Sequence[str], local_bytes: int):
    """As record_collective, but for callers that already know the per-chip byte count.

    Used for whole-pytree collectives, where one logical synchronization covers hundreds of
    arrays and should appear as a single row totalling them rather than hundreds of rows at
    the same source line.
    """
    if not _DCN_AXES:
        return
    crossing = tuple(a for a in axes if a in _DCN_AXES)
    if not crossing:
        return

    site = _caller_site()
    if _DCN_STRICT and not any(site.endswith(s) for s in _DCN_STRICT):
        raise TypeCheckError(
            f"{site}: {op} crosses DCN over {crossing}, which is not in the DCN allowlist. "
            f"Either this collective belongs on ICI (check the mesh hierarchy) or add it to "
            f"set_dcn_allowlist()."
        )

    n = 1
    for a in crossing:
        n *= _AXIS_SIZES.get(a, 1)
    if n <= 1:
        return
    moved = int(local_bytes * _RING_COST[op](n))

    key = (site, op, crossing)
    prev = _DCN_LOG.get(key)
    # Re-tracing the same function must not double-count, but a genuinely bigger tensor at the
    # same site (a different batch size, say) should win.
    if prev is None or moved > prev.bytes_per_chip:
        _DCN_LOG[key] = DcnCollective(site=site, op=op, axes=crossing, bytes_per_chip=moved)


def dcn_collectives() -> list:
    return sorted(_DCN_LOG.values(), key=lambda c: -c.bytes_per_chip)


def dcn_report(bandwidth_gbytes_per_sec: float = None) -> str:
    """Human-readable table of what crosses DCN per chip per step."""
    rows = dcn_collectives()
    if not _DCN_AXES:
        return "COLLECTIVES CROSSING DCN: none (single-slice mesh, no DCN axes declared)"
    lines = [f"COLLECTIVES CROSSING DCN (per chip, per step) over axes {sorted(_DCN_AXES)}"]
    if not rows:
        lines.append("  none -- no collective travels over a DCN axis")
        return "\n".join(lines)
    mib = 1024 * 1024
    for c in rows:
        lines.append(f"  {c.site:<22} {c.op:<13} over {','.join(c.axes):<10} {c.bytes_per_chip / mib:10.1f} MiB")
    total = sum(c.bytes_per_chip for c in rows)
    lines.append(f"  {'-' * 62}")
    lines.append(f"  {'total':<22} {'':<13} {'':<15} {total / mib:10.1f} MiB")
    if bandwidth_gbytes_per_sec:
        secs = total / (bandwidth_gbytes_per_sec * 1e9)
        lines.append(f"  predicted DCN time at {bandwidth_gbytes_per_sec:g} GB/s: {secs * 1e3:.0f} ms")
    return "\n".join(lines)


def is_fully_sharded(spec: jax.sharding.PartitionSpec):
    """True if `spec` is split over every axis weights are meant to be split over.

    DCN axes are excluded from the requirement. Weights are deliberately REPLICATED across
    slices rather than sharded over them -- gathering weights over DCN is the misconfiguration
    this whole hierarchy exists to avoid -- so a weight that names every ICI axis and no DCN
    axis is exactly right, and must not trip the "fully sharded" assertion in the optimizer.

    Reads the mesh from the registry rather than jax's internal trace state, which also means
    this no longer depends on a private jax API.
    """
    axis_count = 0
    for axis in spec:
        if axis is None:
            continue
        elif isinstance(axis, str):
            axis_count += 1
        elif isinstance(axis, tuple):
            axis_count += len(axis)
        else:
            raise ValueError(f"Unknown axis type {axis}")
    return axis_count == len(_AXIS_SIZES) - len(_REPLICA_AXES)
