"""Main training loop, including the model, loss function, and optimizer."""

# Set XLA flags before importing JAX
import init_seqax  # noqa: F401  # isort: skip

import datetime
import math
import operator
import os
import time
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Any, Optional, Tuple, Union

import einops
import hydra
import jax
import jax.numpy as jnp
# from clearml import Task  # lazy-imported below; only needed with a queue
from jax import lax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.tree_util import tree_leaves
from typeguard import typechecked

import jax_extra
import shardlib.shardops as shardops
import shardlib.shardtypes as shardtypes
import slicemon_hook
import training_io
from input_loader import (FlatTokensParams, HuggingFaceDataParams, NanochatDataParams,
                          TokenBatch, TokenBatchParams, get_loader)
from jax_extra import explicit_activation_checkpointing, fold_in_str, save_for_backward
from shardlib.shardtypes import Array, bf16, bool_, f32, make_shardings, pytree_dataclass, u32

shardtypes.register_with_typeguard()
PRNGKey = Any


@dataclass(frozen=True)
class Hparams:
    d_model: int
    n_q_per_kv: int
    n_kv: int
    d_head: int
    layers: int
    vocab: int
    d_ff: int
    rope_max_timescale: int


@pytree_dataclass
class TransformerLayer:
    ln1: f32[b"d_model/t/d"]
    ln2: f32[b"d_model/t/d"]
    w_q: f32[b"d_model/d n_q_per_kv n_kv/t d_head"]
    w_kv: f32[b"2 d_model/d n_kv/t d_head"]
    w_o: f32[b"d_model/d n_q_per_kv n_kv/t d_head"]
    w_gate: f32[b"d_model/d d_ff/t"]
    w_up: f32[b"d_model/d d_ff/t"]
    w_down: f32[b"d_model/d d_ff/t"]


Transformer = Array["layers", TransformerLayer]


@pytree_dataclass
class Model:
    embed: f32[b"vocab/t d_model/d"]
    unembed: f32[b"vocab/t d_model/d"]
    transformer: Transformer
    final_layer_norm: f32[b"d_model/d/t"]

    @staticmethod
    @typechecked
    def init(h: Hparams, rng: PRNGKey) -> "Model":
        embed = jax.random.normal(jax_extra.fold_in_str(rng, "embed"), (h.vocab, h.d_model), dtype=jnp.float32)
        # https://github.com/google/jax/issues/20390 for ones_like with sharding.
        ln1 = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
        ln2 = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
        final_layer_norm = jnp.ones((h.d_model,), dtype=jnp.float32)

        # All of wi/wq/wo/wo/w_kv use truncated_normal initializers with 'fan_in' scaling,
        # i.e. variance set to 1.0/fan_in.
        # The constant is stddev of standard normal truncated to (-2, 2)
        truncated_normal_stddev = 0.87962566103423978

        # scale for tensors with d_model fan_in and truncated normal truncated to (-2, 2)
        d_model_scale = 1 / (math.sqrt(h.d_model) * truncated_normal_stddev)

        w_kv_scale = d_model_scale
        w_q_scale = d_model_scale / math.sqrt(h.d_head)
        total_head_dim = h.n_q_per_kv * h.n_kv * h.d_head
        w_o_scale = 1 / (math.sqrt(total_head_dim) * truncated_normal_stddev)
        w_up_scale = d_model_scale
        w_down_scale = 1 / (math.sqrt(h.d_ff) * truncated_normal_stddev)
        unembed_scale = d_model_scale

        w_q_shape = (h.layers, h.d_model, h.n_q_per_kv, h.n_kv, h.d_head)
        w_q = w_q_scale * jax.random.truncated_normal(fold_in_str(rng, "w_q"), -2, 2, w_q_shape, dtype=jnp.float32)
        w_kv_shape = (h.layers, 2, h.d_model, h.n_kv, h.d_head)
        w_kv = w_kv_scale * jax.random.truncated_normal(fold_in_str(rng, "w_kv"), -2, 2, w_kv_shape, dtype=jnp.float32)
        w_o_shape = w_q_shape
        w_o = w_o_scale * jax.random.truncated_normal(fold_in_str(rng, "w_o"), -2, 2, w_o_shape, dtype=jnp.float32)

        ff_shape = (h.layers, h.d_model, h.d_ff)
        w_gate = w_up_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_gate"), -2, 2, ff_shape, dtype=jnp.float32
        )
        w_up = w_up_scale * jax.random.truncated_normal(fold_in_str(rng, "w_up"), -2, 2, ff_shape, dtype=jnp.float32)
        w_down = w_down_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_down"), -2, 2, ff_shape, dtype=jnp.float32
        )

        unembed = unembed_scale * jax.random.truncated_normal(
            fold_in_str(rng, "unembed"), -2, 2, (h.vocab, h.d_model), dtype=jnp.float32
        )
        arrays = Model(
            embed=embed,
            unembed=unembed,
            transformer=Transformer(
                ln1=ln1,
                ln2=ln2,
                w_q=w_q,
                w_kv=w_kv,
                w_o=w_o,
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
            ),
            final_layer_norm=final_layer_norm,
        )
        shardings = make_shardings(Model)
        return jax.tree.map(lax.with_sharding_constraint, arrays, shardings)

    @typechecked
    def forward_pass(self, h: Hparams, ids: u32[b"B/data/d L"], is_seq_start: bool_[b"B/data/d L"]) -> f32[b"B/data/d L V/t"]:
        ##### Initial embedding lookup.
        embed = shardops.all_gather("V/t M/d -> V/t M", jnp.bfloat16(self.embed))
        x = shardops.index_unreduced("[V/t] M, B/data/d L -> B/data/d L M", embed, ids)
        x = shardops.psum_scatter("B/data/d L M -> B/data/d L M/t", x)

        L = ids.shape[1]
        segment_ids = jnp.cumsum(is_seq_start, axis=1)
        segment_mask: bool_[b"B/data/d L L"] = segment_ids[:, :, jnp.newaxis] == segment_ids[:, jnp.newaxis, :]
        segment_mask: bool_[b"B/data/d L L 1 1"] = segment_mask[
            ..., jnp.newaxis, jnp.newaxis
        ]  # add axes for q_per_k, num_kv_heads dimensions
        causal_mask: bool_[b"1 L L 1 1"] = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_), 0)[
            jnp.newaxis, ..., jnp.newaxis, jnp.newaxis
        ]
        causal_mask: bool_[b"B/data/d L L 1 1"] = jnp.logical_and(segment_mask, causal_mask)

        rope_table = RopeTable.create(L, h)

        ##### Transformer blocks.
        @explicit_activation_checkpointing
        @typechecked
        def loop_body(x: bf16[b"B/data/d L M/t"], layer_weights: TransformerLayer) -> Tuple[bf16[b"B/data/d L M/t"], Tuple[()]]:
            # Pre-attention RMSNorm
            ln1 = shardops.all_gather("M/t/d -> M", jnp.float32(layer_weights.ln1))
            gx = shardops.all_gather("B/data/d L M/t -> B/data/d L M", x)
            nx = jnp.bfloat16(rms_norm(gx) * ln1)

            # Attention, using Grouped Query Attention and RoPE position embeddings.
            w_q = shardops.all_gather("M/d Q K/t D -> M Q K/t D", jnp.bfloat16(layer_weights.w_q))
            q = save_for_backward(shardops.einsum_unreduced("B/data/d L M, M Q K/t D -> B/data/d L Q K/t D", nx, w_q))
            q = rope_table.apply("L D -> 1 L 1 1 D", q)
            w_kv = shardops.all_gather("2 M/d K/t D -> 2 M K/t D", jnp.bfloat16(layer_weights.w_kv))
            k, v = shardops.einsum_unreduced("B/data/d L M, k_v M K/t D -> k_v B/data/d L K/t D", nx, w_kv)
            k = save_for_backward(k)
            v = save_for_backward(v)
            k = rope_table.apply("L d -> 1 L 1 d", k)
            logits = shardops.einsum_unreduced(
                "B/data/d Qlen Q K/t D, B/data/d Klen K/t D -> B/data/d Qlen Klen Q K/t", q, k, preferred_element_type=jnp.float32
            )
            logits = jnp.where(causal_mask, logits, -1e10)
            probs = jnp.bfloat16(jax.nn.softmax(logits, axis=2))
            attn_out = shardops.einsum_unreduced("B/data/d Qlen Klen Q K/t, B/data/d Klen K/t D -> B/data/d Qlen Q K/t D", probs, v)
            w_o = shardops.all_gather("M/d Q K/t D -> M Q K/t D", jnp.bfloat16(layer_weights.w_o))
            attn_out = shardops.einsum_unreduced("B/data/d Qlen Q K/t D, M Q K/t D -> B/data/d Qlen M", attn_out, w_o)
            attn_out = shardops.psum_scatter("B/data/d Qlen M -> B/data/d Qlen M/t", attn_out)
            x = save_for_backward(x + attn_out)

            # Pre-FFN RMSNorm
            ln2 = save_for_backward(shardops.all_gather("M/t/d -> M", jnp.float32(layer_weights.ln2)))
            gx = shardops.all_gather("B/data/d L M/t -> B/data/d L M", x)
            nx = jnp.bfloat16(rms_norm(gx) * ln2)

            # FFN, using SwiGLU
            w_gate = shardops.all_gather("M/d F/t -> M F/t", jnp.bfloat16(layer_weights.w_gate))
            gate_proj = save_for_backward(shardops.einsum_unreduced("B/data/d L M, M F/t -> B/data/d L F/t", nx, w_gate))
            w_up = shardops.all_gather("M/d F/t -> M F/t", jnp.bfloat16(layer_weights.w_up))
            up_proj = save_for_backward(shardops.einsum_unreduced("B/data/d L M, M F/t -> B/data/d L F/t", nx, w_up))
            y = jax.nn.swish(gate_proj) * up_proj
            w_down = shardops.all_gather("M/d F/t -> M F/t", jnp.bfloat16(layer_weights.w_down))
            ffn_out = shardops.einsum_unreduced("B/data/d L F/t, M F/t -> B/data/d L M", y, w_down)
            ffn_out = shardops.psum_scatter("B/data/d L M -> B/data/d L M/t", ffn_out)

            return jnp.bfloat16(x + ffn_out), ()

        x, () = jax.lax.scan(loop_body, jnp.bfloat16(x), self.transformer)

        ##### Final layernorm and output projection.
        x = shardops.all_gather("B/data/d L M/t -> B/data/d L M", x)
        ln = shardops.all_gather("M/t/d -> M", jnp.float32(self.final_layer_norm))
        x = jnp.bfloat16(rms_norm(x) * ln)
        unembed = shardops.all_gather("V/t M/d -> V/t M", jnp.bfloat16(self.unembed))
        logits = shardops.einsum_unreduced(
            "B/data/d L M, V/t M -> B/data/d L V/t", x, unembed, preferred_element_type=jnp.float32
        )

        return logits

    @typechecked
    def loss(self, h: Hparams, batch: TokenBatch) -> f32[b""]:
        # Given sequence-packed targets:
        #   [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
        # we want inputs:
        #   [[0, 1], [0, 3, 4], [0, 6, 7, 8]]
        # which we get by shifting the targets right by 1 and
        # masking sequence-start tokens to 0.
        inputs = jnp.pad(batch.targets[:, :-1], pad_width=((0, 0), (1, 0)))
        is_seq_start: bool_[b"batch/data/d len"] = batch.is_seq_start
        inputs: u32[b"batch/data/d len"] = jnp.where(is_seq_start, 0, inputs)

        logits: f32[b"batch/data/d len V/t"] = self.forward_pass(h, inputs, is_seq_start)
        max_logits: f32[b"batch/data/d len 1"] = lax.pmax(jnp.max(lax.stop_gradient(logits), axis=-1, keepdims=True), "t")
        logits = logits - max_logits
        sum_logits = lax.psum(jnp.sum(jnp.exp(logits), axis=-1, keepdims=True), "t")
        logsumexp = jnp.log(sum_logits)
        logprobs: f32[b"batch/data/d len V/t"] = logits - logsumexp
        logprobs_at_targets = shardops.index_unreduced(
            "batch/data/d len [V/t], batch/data/d len -> batch/data/d len", logprobs, batch.targets
        )
        logprobs_at_targets = shardops.psum_scatter("batch/data/d len -> batch/data/d len/t", logprobs_at_targets)
        tokens_in_global_batch = logprobs_at_targets.size * jax.lax.psum(1, shardtypes.all_axes())
        return -jnp.sum(logprobs_at_targets) / jnp.float32(tokens_in_global_batch)


@pytree_dataclass
class RopeTable:
    sin: f32[b"len d_head2"]
    cos: f32[b"len d_head2"]

    @staticmethod
    def create(max_len: int, hparams: Hparams) -> "RopeTable":
        rope_max_timescale = hparams.rope_max_timescale
        d_head = hparams.d_head
        d = d_head // 2
        # endpoint=False is equivalent to what MaxText does. endpoint=True would be more natural, though.
        timescale = jnp.logspace(0, jnp.log10(jnp.float32(rope_max_timescale)), d, endpoint=False)
        position = jnp.arange(max_len, dtype=jnp.int32)
        sinusoid_inp = jnp.float32(position[:, jnp.newaxis]) / timescale[jnp.newaxis, :]
        sin = jnp.sin(sinusoid_inp)
        cos = jnp.cos(sinusoid_inp)
        return RopeTable(sin=sin, cos=cos)

    def apply(self, rearrange_spec, x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        sin = einops.rearrange(self.sin, rearrange_spec)
        cos = einops.rearrange(self.cos, rearrange_spec)
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin
        return jnp.append(r1, r2, axis=-1)


@typechecked
def rms_norm(x: bf16[b"batch/data/d len M"]) -> bf16[b"batch/data/d len M"]:
    mean2 = save_for_backward(jnp.mean(jax.lax.square(jnp.float32(x)), axis=-1, keepdims=True))
    return jnp.bfloat16(x * jax.lax.rsqrt(mean2 + 1e-6))


@pytree_dataclass
class Metrics:
    loss: f32[b""]
    learning_rate: f32[b""]
    grad_norm: f32[b""]
    raw_grad_norm: f32[b""]


@dataclass(frozen=True)
class TrainingHparams:
    adam_b1: float
    adam_b2: float
    adam_eps: float
    adam_eps_root: float
    weight_decay: float
    warmup_steps: int
    steps: int
    steps_for_lr: int
    cosine_learning_rate_final_fraction: float
    learning_rate: float
    tokens: TokenBatchParams
    seed: int
    queue: Optional[str] = None


# ZeRO-Offload toggle: with SEQAX_OFFLOAD=1, the whole training state (master weights + Adam
# mu/nu) rests in host CPU RAM. Weights are streamed to HBM for the forward/backward pass, and
# the Adam update runs on the host, so optimizer state never occupies HBM.
OFFLOAD = os.environ.get("SEQAX_OFFLOAD") == "1"
try:
    from jax.experimental.compute_on import compute_on
except Exception:
    compute_on = None


@pytree_dataclass
class State:
    weights: Model
    adam_mu: Model
    adam_nu: Model

    @staticmethod
    def init(hparams: Hparams, rng: PRNGKey) -> "State":
        weights = Model.init(hparams, rng)
        adam_mu = jax.tree.map(lambda p: p * 0.0, weights)
        adam_nu = jax.tree.map(lambda p: p * 0.0, weights)
        state = State(weights=weights, adam_mu=adam_mu, adam_nu=adam_nu)
        if OFFLOAD:
            # Park the whole training state in host RAM (pinned_host). `to_host` in shardops is
            # the explicit, in-code analogue of the cross-chip all_gather/psum_scatter.
            state = State(
                weights=jax.tree.map(shardops.to_host, state.weights),
                adam_mu=jax.tree.map(shardops.to_host, state.adam_mu),
                adam_nu=jax.tree.map(shardops.to_host, state.adam_nu),
            )
        return state


def _lr_schedule(step, hparams):
    warmup_lr = (jnp.float32(step) / jnp.float32(hparams.warmup_steps)) * hparams.learning_rate
    cosine = jnp.cos(
        jnp.pi * (jnp.float32(step - hparams.warmup_steps) / jnp.float32(hparams.steps_for_lr - hparams.warmup_steps))
    )
    cosine_lr = hparams.learning_rate * (
        hparams.cosine_learning_rate_final_fraction
        + (1 - hparams.cosine_learning_rate_final_fraction) * (cosine * 0.5 + 0.5)
    )
    return jnp.where(step < hparams.warmup_steps, warmup_lr, cosine_lr)


@partial(jax.jit, static_argnums=(2, 3), donate_argnums=(0,))
def training_step(
    state: State, step: u32[b""], h: Hparams, hparams: TrainingHparams, batch: TokenBatch
) -> Tuple[Any, Metrics]:
    if OFFLOAD:
        # ---- ZeRO-Offload: master weights + Adam mu/nu live on host; update runs on host. ----
        # 1) Stream master weights host -> HBM for the forward/backward pass.
        weights_dev = jax.tree.map(shardops.to_device, state.weights)

        # 2) Forward/backward + global-norm gradient clipping on-device (the norm needs the
        #    cross-chip reduction, hence inside shard_map). Return the *clipped* gradient.
        @partial(shardtypes.typed_shard_map, check_rep=False)
        def grad_step(weights: Model, batch: TokenBatch) -> Tuple[Model, f32[b""], f32[b""]]:
            loss, grad = jax.value_and_grad(lambda w: w.loss(h, batch))(weights)
            loss = jax.lax.psum(loss, shardtypes.all_axes())
            gns = jnp.float32(0.0)
            for g in tree_leaves(grad):
                gns += jnp.sum(jax.lax.square(g))
            global_norm = jnp.sqrt(jax.lax.psum(gns, shardtypes.all_axes()))
            rescale = jnp.minimum(1.0, 1.0 / global_norm)
            grad = jax.tree.map(lambda g: g * rescale, grad)
            return grad, loss, global_norm

        grad_dev, loss, global_norm = grad_step(weights_dev, batch)

        # 3) Move the (clipped) gradient HBM -> host and run AdamW ON THE HOST, so mu/nu and the
        #    updated weights never occupy HBM.
        grad_host = jax.tree.map(shardops.to_host, grad_dev)
        lr = _lr_schedule(step, hparams)
        completed_steps = step + 1
        bc1 = 1 - jnp.float32(hparams.adam_b1) ** completed_steps
        bc2 = 1 - jnp.float32(hparams.adam_b2) ** completed_steps

        def adam_leaf(p, g, mu, nu):
            mu = (1 - hparams.adam_b1) * g + hparams.adam_b1 * mu
            nu = (1 - hparams.adam_b2) * jax.lax.square(g) + hparams.adam_b2 * nu
            u = (mu / bc1) / (jnp.sqrt(nu / bc2 + hparams.adam_eps_root) + hparams.adam_eps)
            u = (u + hparams.weight_decay * p) * lr
            return p - u, mu, nu

        assert compute_on is not None, "SEQAX_OFFLOAD=1 requires jax.experimental.compute_on"
        with compute_on("device_host"):
            new_w, new_mu, new_nu = [], [], []
            for p, g, mu, nu in zip(
                tree_leaves(state.weights),
                tree_leaves(grad_host),
                tree_leaves(state.adam_mu),
                tree_leaves(state.adam_nu),
            ):
                np_, nmu, nnu = adam_leaf(p, g, mu, nu)
                new_w.append(np_)
                new_mu.append(nmu)
                new_nu.append(nnu)

        treedef = jax.tree_util.tree_structure(state.weights)
        # The host update loses the device sharding (host compute yields replicated arrays), so
        # re-apply the correct host sharding (pinned_host + the model's partition spec) to every
        # leaf, so the new state matches the sharding the step was compiled with.
        host_shard = make_shardings(Model, memory_kind="pinned_host")

        def _reshard(leaves):
            tree = jax.tree_util.tree_unflatten(treedef, leaves)
            return jax.tree.map(lambda x, s: jax.lax.with_sharding_constraint(x, s), tree, host_shard)

        new_state = State(
            weights=_reshard(new_w),
            adam_mu=_reshard(new_mu),
            adam_nu=_reshard(new_nu),
        )
        metrics = Metrics(
            loss=loss,
            learning_rate=lr,
            grad_norm=jnp.minimum(global_norm, 1.0),
            raw_grad_norm=global_norm,
        )
        return new_state, metrics

    @partial(
        shardtypes.typed_shard_map, check_rep=False
    )  # check_rep=False for https://github.com/google/jax/issues/20335
    def sharded_step(state: State, step: u32[b""], batch: TokenBatch) -> Tuple[State, Metrics]:
        loss, grad = jax.value_and_grad(lambda weights: weights.loss(h, batch))(state.weights)
        # Gradients have already been reduced across chips because the gradient of the weight `all_gather`
        # is weight-gradient `psum_scatter`. Loss, on the other hand, hasn't been reduced across chips: if we
        # did that inside the autodiff, we'd be double-reducing the loss, effectively multiplying it by the
        # amount of data parallelism.
        #
        # So we reduce the loss across chips _outside_ the autodiff.
        loss = jax.lax.psum(loss, shardtypes.all_axes())

        # ...but that automatic reduction only covers the axes the weights are SHARDED over,
        # because it is the transpose of the weight all_gather. `data` never appears in a weight
        # spec -- weights are replicated across slices, deliberately, so that gathering them
        # stays on the fast network -- so nothing has yet combined the gradients that different
        # slices computed from different batch shards. This is that step, and on a multi-slice
        # mesh it is the single collective in the whole training step that crosses DCN.
        #
        # It is a no-op when `data` has size 1, which is what makes the single-slice path
        # identical to what it was before any of this existed.
        grad = shardops.all_reduce_tree(grad, over="data")

        # Other than global-norm of gradients, no other communication is needed during the weight update,
        # because weights and grads are already fully sharded, as checked below.

        # Calculate learning rate from step number.
        # We use linear warmup then cosine decay. See https://arxiv.org/pdf/2307.09288.pdf section 2.2
        warmup_lr = (jnp.float32(step) / jnp.float32(hparams.warmup_steps)) * hparams.learning_rate
        cosine = jnp.cos(
            jnp.pi
            * (jnp.float32(step - hparams.warmup_steps) / jnp.float32(hparams.steps_for_lr - hparams.warmup_steps))
        )
        cosine_lr = hparams.learning_rate * (
            hparams.cosine_learning_rate_final_fraction
            + (1 - hparams.cosine_learning_rate_final_fraction) * (cosine * 0.5 + 0.5)
        )
        lr = jnp.where(step < hparams.warmup_steps, warmup_lr, cosine_lr)

        # AdamW optimizer with global gradient clipping.
        grad_leaves, grad_treedef = jax.tree_util.tree_flatten(grad)
        global_norm_square = jnp.float32(0.0)
        for g in grad_leaves:
            assert g.dtype == jnp.float32
            global_norm_square += jnp.sum(jax.lax.square(g))
        global_norm_square = jax.lax.psum(global_norm_square, shardtypes.all_axes())
        global_norm = jnp.sqrt(global_norm_square)
        rescale = jnp.minimum(1.0, 1.0 / global_norm)

        new_ps = []
        new_mus = []
        new_nus = []
        for p, g, mu, nu, spec in zip(
            tree_leaves(state.weights),
            grad_leaves,
            tree_leaves(state.adam_mu),
            tree_leaves(state.adam_nu),
            tree_leaves(shardtypes.make_partition_specs(State)),
        ):
            assert shardtypes.is_fully_sharded(spec), (
                "Weight update is only correctly scaled for fully sharded weights."
            )
            # Gradient clipping
            g = g * rescale
            # Adam scaling
            mu = (1 - hparams.adam_b1) * g + hparams.adam_b1 * mu
            nu = (1 - hparams.adam_b2) * jax.lax.square(g) + hparams.adam_b2 * nu
            # We need step numbers to start at 1, not 0. Otherwise the bias correction produces NaN.
            completed_steps = step + 1
            mu_hat = mu / (1 - jnp.float32(hparams.adam_b1) ** completed_steps)
            nu_hat = nu / (1 - jnp.float32(hparams.adam_b2) ** completed_steps)
            g = mu_hat / (jnp.sqrt(nu_hat + hparams.adam_eps_root) + hparams.adam_eps)
            # Weight decay
            g += hparams.weight_decay * p
            # Learning rate
            g *= lr

            # Apply update
            new_ps.append(p - g)
            new_mus.append(mu)
            new_nus.append(nu)

        new_state = State(
            weights=jax.tree_util.tree_unflatten(grad_treedef, new_ps),
            adam_mu=jax.tree_util.tree_unflatten(grad_treedef, new_mus),
            adam_nu=jax.tree_util.tree_unflatten(grad_treedef, new_nus),
        )
        metrics = Metrics(
            loss=loss,
            learning_rate=lr,
            grad_norm=global_norm * rescale,
            raw_grad_norm=global_norm,
        )
        return new_state, metrics

    return sharded_step(state, step, batch)


@dataclass(frozen=True)
class Paths:
    root_working_dir: str
    model_name: str
    # Directory for XLA's persistent compilation cache. Compiling the training step takes
    # tens of seconds to minutes, and every process of every run repeats that work, because
    # the compiler's output is only kept in the process's memory. Pointing this at shared
    # storage (e.g. the same GCS bucket as root_working_dir) makes the first process to
    # compile a given (HLO, device kind, XLA flags) publish its result, and every later
    # process -- the other hosts in this run, and every subsequent run of the same config --
    # load it instead of recompiling. None disables the cache.
    compilation_cache_dir: Optional[str] = None


@dataclass(frozen=True)
class MeshConfig:
    """Mesh axes, split by which physical network carries them.

    `d` and `t` are ICI axes: chips within one slice, connected by the fast interconnect.

    `dcn` names the axes carried by the datacenter network *between* slices, as a
    comma-separated "name:size" list. It is deliberately a string rather than a nested dict
    so it survives a Hydra command-line override unchanged:

        mesh.dcn=data:2          two slices, one DCN axis, gradients replicated across them
        mesh.dcn=outer:2,inner:2 four slices arranged as a two-level DCN hierarchy
        (unset)                  one slice -- exactly the previous behavior, no DCN at all

    The total mesh is the product: with d=8, t=1 and dcn=data:2 that is 8 chips per slice
    across 2 slices, 16 chips, and the `data` axis is the only one whose collectives pay
    DCN bandwidth.
    """

    d: int
    t: int
    dcn: Optional[str] = None


@dataclass(frozen=True)
class Config:
    model: Hparams
    training: TrainingHparams
    paths: Paths
    num_hosts: int
    checkpoint_interval: int
    mesh: MeshConfig
    io: training_io.IOConfig
    flat_tokens: Optional[FlatTokensParams] = None
    hf_dataset: Optional[HuggingFaceDataParams] = None
    nanochat: Optional[NanochatDataParams] = None

    def __post_init__(self):
        sources = [self.flat_tokens, self.hf_dataset, self.nanochat]
        provided = [s for s in sources if s is not None]
        assert provided, "Must provide one of flat_tokens, hf_dataset, or nanochat."
        assert len(provided) == 1, (
            "Should specify exactly one of flat_tokens, hf_dataset, or nanochat."
        )

    @cached_property
    def training_data(self) -> Union[FlatTokensParams, HuggingFaceDataParams, NanochatDataParams]:
        return self.flat_tokens or self.hf_dataset or self.nanochat


def _parse_dcn(spec: Optional[str]) -> dict:
    """Parse "data:2" or "outer:2,inner:2" into an ordered {name: size}."""
    if not spec:
        return {}
    result = {}
    for piece in spec.split(","):
        name, sep, size = piece.partition(":")
        name = name.strip()
        if not name or not sep or not size.strip():
            raise ValueError(f"Bad mesh.dcn entry {piece!r}: expected 'name:size', e.g. 'data:2'")
        if name not in ("data", "d"):
            raise ValueError(
                f"mesh.dcn axis {name!r} is not supported. Use 'data:N' for the correct "
                f"hierarchy (weights replicated across slices, only gradients cross DCN), or "
                f"'d:N' to deliberately stretch the FSDP axis across slices -- the WRONG "
                f"hierarchy, which puts weight gathering on the slow link. 'd:N' exists only "
                f"to measure how bad that is; it is never what you want in production."
            )
        result[name] = int(size)
    return result


def build_mesh(mesh_config: MeshConfig) -> Mesh:
    """Build the device mesh, and tell shardtypes which of its axes are DCN.

    With no `dcn` this is exactly the previous behavior -- create_device_mesh over ("d", "t")
    -- so a single-slice config is bit-for-bit the same program it was before this existed.

    With `dcn`, the mesh is built by create_hybrid_device_mesh, which places chips within a
    slice along the ICI axes and slices along the DCN axes. That function multiplies its two
    shapes elementwise, so both must have the same length: every axis is carried by exactly
    one network and takes a 1 in the other. DCN axes come first, matching the physical
    hierarchy -- outermost/slowest first.
    """
    dcn = _parse_dcn(mesh_config.dcn)
    replicas = dcn.get("data", 1)
    d_stretch = dcn.get("d", 1)

    # `data` exists in every configuration, size 1 when there is one slice. The shape specs say
    # `B/data/d` unconditionally, so the axis has to be there for them to resolve; a size-1
    # leading axis shards nothing and leaves the single-slice layout exactly as it was.
    names = ("data", "d", "t")
    ici_shape = (1, mesh_config.d, mesh_config.t)
    # Each axis gets its slice-spanning factor here. `data` is the correct place for it.
    # `d` is the wrong place, deliberately: it stretches the FSDP axis across slices so weight
    # gathering has to cross DCN. That is arm 2.
    dcn_shape = (replicas, d_stretch, 1)
    expected = math.prod(ici_shape) * math.prod(dcn_shape)
    avail = jax.device_count()
    if expected > avail:
        raise ValueError(
            f"mesh data={replicas} d={mesh_config.d} t={mesh_config.t} needs {expected} devices "
            f"but only {avail} are visible"
        )
    # A mesh smaller than the allocation is allowed, for chip-count scaling studies: hold
    # tokens-per-chip fixed and vary how many chips participate. This only works on a
    # SINGLE-HOST slice. On multi-host, the hosts whose chips are excluded still have to join
    # every computation and fail with "Device assignment (Computations: N Replicas: 1...)" -
    # verified on a 4-host v4-32. So the smallest usable unit there is the whole slice.
    use = jax.devices()[:expected]
    if expected < avail:
        print(f"[mesh] using {expected} of {avail} visible devices (chip-count scaling study)")

    if replicas == 1 and d_stretch == 1:
        devices = mesh_utils.create_device_mesh(list(ici_shape), use)
        mesh = Mesh(devices, names)
        # Replicated-over but not DCN-carried: with one slice there is no slow network to
        # account for, yet weights are still not sharded over `data`.
        shardtypes.declare_mesh(mesh, dcn_axes=(), replica_axes=("data",))
        return mesh

    devices = mesh_utils.create_hybrid_device_mesh(ici_shape, dcn_shape, use)
    mesh = Mesh(devices, names)
    dcn_axes = tuple(n for n, f in zip(names, dcn_shape) if f > 1)
    shardtypes.declare_mesh(mesh, dcn_axes=dcn_axes, replica_axes=("data",))
    if d_stretch > 1:
        print(
            f"[mesh] WARNING: d spans {d_stretch} slices ({mesh_config.d} chips in-slice x "
            f"{d_stretch}). Weight gathering will cross DCN. This is the wrong hierarchy and is "
            f"only useful as a measurement. Byte counts below are an UPPER BOUND: the report "
            f"charges every hop of a mixed ICI/DCN ring at DCN rate, when only the inter-slice "
            f"hops actually pay it."
        )
    return mesh


def main_contained(config, logger):
    """Main program, which does not access external services except as specified by config.paths or logger."""
    # Use partitionable (and hopefully fusable!) RNG.
    #
    # This is slower in compute time than 'unsafe_rbg' with flag '--xla_tpu_spmd_rng_bit_generator_unsafe=true',
    # but hopefully faster in memory time because it's fusable.
    # TODO: check this is true and if not, provide our own that actually is fusable.
    jax.config.update("jax_threefry_partitionable", True)

    # Reuse compiled programs across processes and across runs. Must be set before the first
    # compilation below. We only cache programs that took over a second to compile, so that
    # the cache holds the training step rather than the many trivial programs around it.
    if config.paths.compilation_cache_dir is not None:
        jax.config.update("jax_compilation_cache_dir", config.paths.compilation_cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        print(f"[compile] persistent compilation cache: {config.paths.compilation_cache_dir}")

    with build_mesh(config.mesh):
        root_rng = jax.random.PRNGKey(config.training.seed)

        loader = get_loader("train", config.training_data, config.training.tokens)
        assert config.model.vocab > loader.max_token_id, f"{config.model.vocab} vs {loader.max_token_id}"

        model_dir = os.path.join(config.paths.root_working_dir, config.paths.model_name)
        training_io.mkdir(model_dir)
        state = jax.jit(partial(State.init, config.model))(fold_in_str(root_rng, "init"))
        state, start_step = training_io.load_checkpoint_if_it_exists(model_dir, state, config.io)

        # Explicitly compile training step, to record XLA HLO graph.
        # See https://bnikolic.co.uk/blog/python/jax/2022/02/22/jax-outputgraph-rev
        c_training_step = training_step.lower(
            state, jnp.uint32(0), config.model, config.training, loader.load(0)
        ).compile()
        # Print the compiled step's memory estimate right after compile, BEFORE any step runs,
        # so it is captured even for configs that OOM at runtime. Lining args+temp up against the
        # empirical OOM boundary reconciles the (over-counting) upper bound against ground truth.
        try:
            _ma = c_training_step.memory_analysis()
            _gib = 2**30
            _args = _ma.argument_size_in_bytes / _gib
            _temp = _ma.temp_size_in_bytes / _gib
            print(
                f"[compile] per-device memory estimate: args={_args:.3f} GiB, temp={_temp:.3f} GiB, "
                f"args+temp(upper bound)={_args + _temp:.3f} GiB (HBM limit ~30.75 GiB)"
            )
        except Exception as _e:
            print(f"[compile] memory_analysis unavailable: {_e}")

        # What this step pushes across the slow network, known before a single step has run.
        # Two things make this worth printing rather than reading off a profile afterwards:
        # it catches a mesh whose hierarchy does not match the network (weight gathering on
        # DCN) at trace time instead of after a wasted run, and it supplies the numerator for
        # a bandwidth measurement -- a profiler saying a collective took 4.2 ms means nothing
        # until you know how many bytes it moved.
        print(shardtypes.dcn_report(training_io.get_dcn_bandwidth_per_chip()))
        # Publish training-loop metrics for slicemon. Chip metrics can't tell you
        # whether the work is useful -- MFU needs FLOPs/step and step time, which
        # only live here.
        slicemon_hook.configure(
            model_name=config.paths.model_name,
            params=jax.tree.reduce(operator.add, jax.tree.map(lambda w: w.size, state.weights)),
            tokens_per_step=loader.load(start_step).targets.size,
            device_flops=training_io.get_flops_per_device(),
            num_devices=jax.device_count(),
            mesh_d=config.mesh.d,
            mesh_t=config.mesh.t,
            total_steps=config.training.steps,
            seq_len=config.training.tokens.len,
            batch=config.training.tokens.batch,
        )

        date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        pass  # training_io.save_hlo_svg(os.path.join(model_dir, f"training_step_optimized_hlo_{date}.svg"), c_training_step)

        for step in range(start_step, config.training.steps):
            if step % config.checkpoint_interval == 0 and step > start_step:
                training_io.save_checkpoint(model_dir, step, state, config.io)

            # We profile on the second step, because the first step has a long pause for XLA
            # compilation and initial shuffle buffer loading.
            #
            # The trace capture itself is opt-in, because it KILLS multi-host runs. Only
            # process 0 enters this block, and stop_profile below spends ~30s writing perfetto
            # traces while every other host sits waiting in a collective; the slice then aborts
            # with "The program continuator has halted unexpectedly" at step 3. Single-host runs
            # survive it, which is why it went unnoticed.
            #
            # The time.time() measurement stays unconditional -- it is what the MFU line reads,
            # and it costs nothing. Set SEQAX_PROFILE=1 to capture an actual trace, which is
            # only safe on a single-host slice.
            _want_trace = os.environ.get("SEQAX_PROFILE") == "1"
            if _want_trace and jax.process_index() == 0 and step == start_step + 1:
                jax.block_until_ready(state)
                training_io.start_profile()
                profile_start = time.time()

            state, output = c_training_step(state, jnp.uint32(step), loader.load(step))

            # Run profile for two steps, to include data loading time in between them.
            #
            # This entire block is process-0-only and sits between two steps: block_until_ready,
            # trace capture, memory_analysis, memory_stats, several prints. On a single host that
            # is harmless. On a multi-host slice it is fatal - process 0 stops enqueuing while
            # the other hosts sit in collectives, and the program dies at the next step with
            # "The program continuator has halted unexpectedly". Gating only the trace capture
            # was not enough; the whole asymmetry has to go.
            #
            # So it is opt-in via SEQAX_PROFILE=1, safe on a single-host slice. For multi-host,
            # take step time from the per-step log timestamps instead (tools/step_times.py),
            # which is both survivable and more accurate than this two-step sample.
            if _want_trace and jax.process_index() == 0 and step == start_step + 2:
                jax.block_until_ready(state)
                profile_duration = time.time() - profile_start
                training_io.stop_profile(model_dir)

                # Print MFU, including (one step of) data loading time.
                print(f"Profile time: {profile_duration}s for 2 steps.")
                model_params = jax.tree.reduce(operator.add, jax.tree.map(lambda w: w.size, state.weights))
                tokens = loader.load(step).targets.size
                print(f"Model params: {model_params:_}")
                print(f"Tokens: {tokens:_}")
                device_flops = training_io.get_flops_per_device()
                num_devices = jax.device_count()
                print(
                    f"MFU (projections only): {100 * (2 * 6 * model_params * tokens / (num_devices * profile_duration)) / device_flops:.2f}% MFU"
                )

                # Phase 0 instrumentation (optimizer-state host-offload baseline).
                # Report the memory footprint of each State component and peak device HBM.
                # Motivation: Adam's mu/nu are each ~1x the weights, so optimizer state is
                # ~2x the model and is the primary target for host offload.
                def _tree_bytes(tree):
                    return jax.tree.reduce(operator.add, jax.tree.map(lambda a: a.size * a.dtype.itemsize, tree))

                gib = 2**30
                weight_bytes = _tree_bytes(state.weights)
                mu_bytes = _tree_bytes(state.adam_mu)
                nu_bytes = _tree_bytes(state.adam_nu)
                print(
                    f"State footprint (global): weights={weight_bytes / gib:.3f} GiB, "
                    f"adam_mu={mu_bytes / gib:.3f} GiB, adam_nu={nu_bytes / gib:.3f} GiB "
                    f"(optimizer state = {(mu_bytes + nu_bytes) / max(weight_bytes, 1):.2f}x weights)"
                )
                try:
                    mem = jax.local_devices()[0].memory_stats()
                except Exception:
                    mem = None
                if mem and "peak_bytes_in_use" in mem:
                    peak_gib = mem["peak_bytes_in_use"] / gib
                    limit_gib = mem.get("bytes_limit", 0) / gib
                    print(
                        f"Peak HBM per device: {peak_gib:.3f} GiB"
                        + (f" / {limit_gib:.3f} GiB limit" if limit_gib else "")
                    )
                else:
                    print("Peak HBM per device: unavailable (backend has no memory_stats; expected on CPU)")

                # True per-device training memory, read from XLA's buffer assignment.
                # `peak_bytes_in_use` above reports resident state (weights + optimizer) but omits the
                # backward-pass activation/gradient scratch, so it under-reports the real training peak
                # (e.g. 2b shows ~6 GiB yet a 4b model OOMs). `memory_analysis()` reads the compiler's
                # actual buffer assignment, whose temp_size_in_bytes captures that scratch.
                try:
                    ma = c_training_step.memory_analysis()
                except Exception:
                    ma = None
                if ma is not None:
                    arg_gib = ma.argument_size_in_bytes / gib
                    temp_gib = ma.temp_size_in_bytes / gib
                    out_gib = ma.output_size_in_bytes / gib
                    alias_gib = ma.alias_size_in_bytes / gib
                    # Args and donated outputs share buffers (alias), so subtract the overlap.
                    true_peak_gib = arg_gib + temp_gib + out_gib - alias_gib
                    print(
                        f"Compiled memory (per device): args(weights+opt+in)={arg_gib:.3f} GiB, "
                        f"temp(activations+grad scratch)={temp_gib:.3f} GiB, "
                        f"output={out_gib:.3f} GiB, alias={alias_gib:.3f} GiB"
                    )
                    print(f"True peak training memory per device: {true_peak_gib:.3f} GiB")
                else:
                    print("Compiled memory analysis: unavailable on this backend/JAX version")

            training_io.log(step, logger, output)


@hydra.main(config_path="configs", version_base=None)
def main(config):
    config = jax_extra.make_dataclass_from_dict(Config, config)
    if config.training.queue:
        from clearml import Task
        task = Task.init(project_name="testing", task_name=config.paths.model_name)
        logger = task.get_logger()
        task.execute_remotely(queue_name=config.training.queue)
        task.launch_multi_node(config.num_hosts, wait=True)
        if int(os.environ["RANK"]) > 0:
            task.set_system_tags((task.get_system_tags() or []) + ["hidden"])
        jax.distributed.initialize(
            os.environ["MASTER_ADDR"] + ":" + os.environ["MASTER_PORT"],
            num_processes=int(os.environ["WORLD_SIZE"]),
            process_id=int(os.environ["RANK"]),
        )
    else:
        logger = None
    main_contained(config, logger)


if __name__ == "__main__":
    main()
