"""Publishes training-loop metrics for slicemon, from inside the seqax train loop.

Chip metrics (duty cycle, HBM, ICI) come from libtpu and say whether the silicon
is busy. They cannot say whether the work is *useful* -- that needs FLOPs/step
and step time, which only the training process knows. This writes those to a
local JSON that host_agent.py picks up and publishes alongside the chip metrics,
so the monitor can show MFU next to duty cycle.

Two insertions, both tiny:

    # train.py, in main_contained() just before the step loop
    slicemon_hook.configure(
        model_name=config.paths.model_name,
        params=jax.tree.reduce(operator.add,
                               jax.tree.map(lambda w: w.size, state.weights)),
        tokens_per_step=loader.load(start_step).targets.size,
        device_flops=training_io.get_flops_per_device(),
        num_devices=jax.device_count(),
        mesh_d=config.mesh.d, mesh_t=config.mesh.t,
        total_steps=config.training.steps,
    )

    # training_io.py, at the end of log()
    slicemon_hook.record(step, metrics_dict)

Writes are throttled and atomic, so leaving this on for a multi-day run costs
nothing measurable.
"""

import json
import math
import os
import sys
import tempfile
import time
from collections import deque

DEFAULT_PATH = "/tmp/slicemon-train.json"

# jaxformer reached 3.28 nats on cl100k (~4.0 bytes/token) = 1.183 bits/byte.
# Measured on this corpus, our 32,768-vocab tokenizer averages 4.693 bytes/token,
# so the equivalent target here is 1.183 * ln(2) * 4.693 = 3.85 nats. Comparing
# raw loss across tokenizers is meaningless; bits/byte is the portable unit.
DEFAULT_TARGET_LOSS = 3.85
DEFAULT_BYTES_PER_TOKEN = 4.693
TARGET_BPB = 1.183

_cfg = {}
_state = {
    "last_write": 0.0,
    "last_step_ts": None,
    "step_times": deque(maxlen=200),
    "curve": deque(maxlen=400),   # (cumulative_tokens, loss) for extrapolation
    "loss_hist": deque(maxlen=120),
    "start_ts": None,
    "start_step": None,
}


def configure(*, model_name="run", params=0, tokens_per_step=0, device_flops=0.0,
              num_devices=1, mesh_d=1, mesh_t=1, total_steps=0,
              target_loss=DEFAULT_TARGET_LOSS, bytes_per_token=DEFAULT_BYTES_PER_TOKEN,
              path=None, min_interval=2.0, seq_len=0, batch=0):
    _cfg.update(
        model_name=model_name, params=int(params),
        tokens_per_step=int(tokens_per_step), device_flops=float(device_flops),
        num_devices=int(num_devices), mesh_d=int(mesh_d), mesh_t=int(mesh_t),
        total_steps=int(total_steps), target_loss=float(target_loss),
        bytes_per_token=float(bytes_per_token), seq_len=int(seq_len), batch=int(batch),
        path=path or os.environ.get("SLICEMON_TRAIN_PATH", DEFAULT_PATH),
        min_interval=float(min_interval),
    )
    _state["start_ts"] = time.time()
    _init_wandb()


def _init_wandb():
    """Opt-in via WANDB_RUN. Process 0 only -- 8 hosts logging is 8 duplicate runs.

    Never let logging break a two-day job: any failure here degrades to no W&B
    rather than raising into the training loop.
    """
    run_name = os.environ.get("WANDB_RUN")
    if not run_name or run_name == "dummy":
        return
    try:
        import jax
        if jax.process_index() != 0:
            return
    except Exception:
        pass
    try:
        import wandb
    except ImportError:
        sys.stderr.write("[slicemon] WANDB_RUN set but wandb not installed; "
                         "pip install wandb\n")
        return
    try:
        _state["wandb"] = wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "seqax-v4-32"),
            name=run_name,
            config={
                "model_name": _cfg["model_name"],
                "params": _cfg["params"],
                "mesh_d": _cfg["mesh_d"],
                "mesh_t": _cfg["mesh_t"],
                "num_devices": _cfg["num_devices"],
                "seq_len": _cfg["seq_len"],
                "batch": _cfg["batch"],
                "tokens_per_step": _cfg["tokens_per_step"],
                "total_steps": _cfg["total_steps"],
                "device_flops": _cfg["device_flops"],
                "bytes_per_token": _cfg["bytes_per_token"],
                "target_loss": _cfg["target_loss"],
            },
        )
        print(f"[slicemon] wandb run '{run_name}' started", flush=True)
    except Exception as exc:
        sys.stderr.write(f"[slicemon] wandb init failed: {exc}\n")
        _state["wandb"] = None


def _peak_hbm_gib():
    try:
        import jax
        return jax.local_devices()[0].memory_stats()["peak_bytes_in_use"] / (1 << 30)
    except Exception:
        return None


def bits_per_byte(loss_nats, bytes_per_token):
    """Cross-entropy normalized out of the tokenizer.

    Per-token loss is not comparable between tokenizers -- a vocabulary that
    compresses better packs more information into each token and so reports a
    larger loss for identical modeling quality. Dividing by bytes per token (and
    converting nats to bits) gives a number that is.
    """
    if loss_nats is None or not bytes_per_token:
        return None
    return loss_nats / (math.log(2) * bytes_per_token)


def _pick(metrics, name):
    """training_io builds keys via keystr, e.g. "['loss']". Match loosely."""
    for key, value in metrics.items():
        if name == "".join(c for c in key if c.isalnum() or c == "_").strip("_"):
            return value
    for key, value in metrics.items():
        if name in key:
            return value
    return None


def _percentile(sorted_values, q):
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def _eta_to_target(curve, target, tokens_per_sec):
    """Fit loss against log10(tokens) over recent history and extrapolate.

    Loss falls roughly linearly in log(tokens) over a run's middle, which is good
    enough for an ETA and honest about being an extrapolation, not a prediction.
    """
    pts = [(math.log10(t), l) for t, l in curve if t > 0 and l is not None and l > 0]
    if len(pts) < 12 or not tokens_per_sec:
        return None
    pts = pts[-200:]
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    if sxx <= 0 or sxy >= 0:      # not descending -> no meaningful ETA
        return None
    slope = sxy / sxx
    intercept = my - slope * mx
    log_target = (target - intercept) / slope
    if not (0 < log_target < 15):
        return None
    tokens_needed = 10 ** log_target
    current = 10 ** pts[-1][0]
    if tokens_needed <= current:
        return 0.0
    return (tokens_needed - current) / tokens_per_sec


def record(step, metrics):
    """Called once per step from training_io.log(). Cheap and throttled."""
    if not _cfg:
        return
    now = time.time()
    st = _state

    if st["last_step_ts"] is not None:
        dt = now - st["last_step_ts"]
        # Step 0 includes XLA compilation; ignore anything wildly off-scale.
        if 0.0 < dt < 3600:
            st["step_times"].append(dt)
    else:
        st["start_step"] = step
    st["last_step_ts"] = now

    loss = _pick(metrics, "loss")
    tokens = (step + 1) * _cfg["tokens_per_step"]
    if loss is not None:
        st["curve"].append((tokens, loss))
        st["loss_hist"].append(loss)

    # Per-step scalars go to W&B at full resolution -- these are free to compute
    # and loss-curve detail is the whole point. Derived metrics ride along on the
    # throttled path below.
    wb = st.get("wandb")
    if wb is not None:
        try:
            wb.log({
                "train/loss": loss,
                "train/bits_per_byte": bits_per_byte(loss, _cfg["bytes_per_token"]),
                "train/grad_norm": _pick(metrics, "grad_norm"),
                "train/raw_grad_norm": _pick(metrics, "raw_grad_norm"),
                "train/lr": _pick(metrics, "learning_rate"),
                "progress/tokens": tokens,
                "progress/tokens_billions": tokens / 1e9,
                "progress/epoch_frac": (step / _cfg["total_steps"]
                                        if _cfg["total_steps"] else None),
            }, step=step)
        except Exception:
            pass

    if now - st["last_write"] < _cfg["min_interval"]:
        return
    st["last_write"] = now

    times = sorted(st["step_times"])
    # Drop the compile outlier before summarizing.
    steady = times[:-1] if len(times) > 4 else times
    p50 = _percentile(steady, 0.50)
    p99 = _percentile(steady, 0.99)

    mfu = tok_per_sec = flops_per_chip = None
    if p50 and p50 > 0 and _cfg["params"] and _cfg["device_flops"]:
        # 6ND, matching seqax's own "projections only" MFU printout.
        flops_per_step = 6.0 * _cfg["params"] * _cfg["tokens_per_step"]
        rate = flops_per_step / p50
        flops_per_chip = rate / _cfg["num_devices"]
        mfu = 100.0 * flops_per_chip / _cfg["device_flops"]
        tok_per_sec = _cfg["tokens_per_step"] / p50

    remaining = None
    if p50 and _cfg["total_steps"]:
        remaining = max(0, _cfg["total_steps"] - step) * p50

    payload = {
        "kind": "slicemon-train",
        "ts": now,
        "model_name": _cfg["model_name"],
        "step": int(step),
        "total_steps": _cfg["total_steps"],
        "tokens": int(tokens),
        "tokens_per_step": _cfg["tokens_per_step"],
        "params": _cfg["params"],
        "mesh": {"d": _cfg["mesh_d"], "t": _cfg["mesh_t"]},
        "loss": loss,
        "bpb": bits_per_byte(loss, _cfg["bytes_per_token"]),
        "bytes_per_token": _cfg["bytes_per_token"],
        "target_bpb": TARGET_BPB,
        "lr": _pick(metrics, "learning_rate"),
        "grad_norm": _pick(metrics, "grad_norm"),
        "raw_grad_norm": _pick(metrics, "raw_grad_norm"),
        "step_time": p50,
        "step_p99": p99,
        "mfu": mfu,
        "tflops_per_chip": (flops_per_chip / 1e12) if flops_per_chip else None,
        "tokens_per_sec": tok_per_sec,
        "target_loss": _cfg["target_loss"],
        "eta_target_s": _eta_to_target(st["curve"], _cfg["target_loss"], tok_per_sec),
        "eta_done_s": remaining,
        "elapsed_s": now - (st["start_ts"] or now),
        "loss_hist": list(st["loss_hist"]),
    }

    if wb is not None:
        try:
            wb.log({
                "perf/mfu_pct": mfu,
                "perf/step_time_s": p50,
                "perf/step_time_p99_s": p99,
                "perf/step_jitter_pct": ((p99 - p50) / p50 * 100) if p50 and p99 else None,
                "perf/tflops_per_chip": (flops_per_chip / 1e12) if flops_per_chip else None,
                "perf/tokens_per_sec": tok_per_sec,
                "perf/peak_hbm_gib": _peak_hbm_gib(),
                "perf/elapsed_hours": (now - (st["start_ts"] or now)) / 3600,
                "eta/hours_to_target": (payload_eta / 3600
                                        if (payload_eta := _eta_to_target(
                                            st["curve"], _cfg["target_loss"],
                                            tok_per_sec)) is not None else None),
                "eta/hours_remaining": (remaining / 3600) if remaining else None,
            }, step=step)
        except Exception:
            pass

    path = _cfg["path"]
    try:
        directory = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".slicemon-", suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        os.replace(tmp, path)
    except Exception:
        pass  # never let telemetry take down a two-day run
