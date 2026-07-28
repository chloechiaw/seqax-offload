#!/usr/bin/env python3
"""Per-host TPU metrics agent.

Runs on every worker of a TPU slice. Reads per-chip metrics out of libtpu's
TPU Monitoring Library and publishes one small JSON blob per host to GCS (or a
local directory) on a fixed interval. slice_monitor.py on your laptop reads
those blobs and renders the whole slice.

    python3 host_agent.py --bucket my-bucket --slice v4-ondemand

Notes:
  * Chip metrics only exist while a workload (JAX / PyTorch-XLA) holds the
    devices. With no job running you get liveness and HBM totals but duty cycle
    and TensorCore utilization will be absent. That is expected, not a failure.
  * libtpu has renamed these metrics between releases, so every metric is
    looked up through a candidate list. If chip values stay empty during a real
    run, print list_supported_metrics() on a worker and add the name you see to
    the matching tuple below.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

AGENT_VERSION = 1

METADATA_ROOT = "http://metadata.google.internal/computeMetadata/v1"
METADATA_HEADERS = {"Metadata-Flavor": "Google"}

# Candidate metric names, most current first. See module docstring.
DUTY_CYCLE_NAMES = ("duty_cycle_pct", "duty_cycle_percent", "duty_cycle")
HBM_USED_NAMES = ("hbm_capacity_usage", "hbm_usage", "memory_usage")
HBM_TOTAL_NAMES = ("hbm_capacity_total", "hbm_total", "memory_total")
TENSORCORE_NAMES = ("tensorcore_util", "tensorcore_utilization", "tensorcore_util_pct")
# ICI / collective timings. With FSDP the slice all-gathers params and
# reduce-scatters grads every step, so this is where a bad link shows up.
COLLECTIVE_NAMES = ("collective_e2e_latency", "collective_e2e_latency_us",
                    "all_reduce_latency", "collective_latency")
# Host<->device transfer. High here during a run on real data means the input
# pipeline is starving the chips, which no loss curve will ever show you.
TRANSFER_NAMES = ("buffer_transfer_latency", "host_to_device_transfer_latency",
                  "device_to_host_transfer_latency")

# Written by slicemon_hook.py from inside the seqax training loop.
TRAIN_SIDECAR = os.environ.get("SLICEMON_TRAIN_PATH", "/tmp/slicemon-train.json")

GIB = float(1 << 30)


# --------------------------------------------------------------------------
# libtpu metric plumbing
# --------------------------------------------------------------------------

def import_tpumonitoring():
    """Return the tpumonitoring module, or None if libtpu isn't importable."""
    try:
        from libtpu.sdk import tpumonitoring  # type: ignore
        return tpumonitoring
    except Exception:
        pass
    try:
        from libtpu import sdk  # type: ignore
        return sdk.tpumonitoring
    except Exception:
        return None


def supported_metrics(mon):
    try:
        names = mon.list_supported_metrics()
    except Exception:
        return set()
    out = set()
    for n in names or ():
        # Depending on version this is a list of strings or of objects.
        out.add(n if isinstance(n, str) else getattr(n, "name", str(n)))
    return out


def _coerce(value):
    """Best-effort numeric coercion of one metric datapoint."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().rstrip("%"))
        except ValueError:
            return None
    for attr in ("value", "val", "data"):
        if hasattr(value, attr):
            inner = getattr(value, attr)
            if not callable(inner):
                return _coerce(inner)
    return None


def read_metric(mon, names, available):
    """Read the first available metric from `names` as a list ordered by device.

    Returns (list_of_floats_or_None, resolved_name). libtpu has returned this
    payload as a dict keyed by device id, as a list of (device, value) pairs,
    and as a bare list, depending on version -- all three are handled.
    """
    for name in names:
        if available and name not in available:
            continue
        try:
            metric = mon.get_metric(name)
        except Exception:
            continue
        data = metric
        if hasattr(metric, "data"):
            try:
                data = metric.data()
            except Exception:
                continue

        values = None
        if isinstance(data, dict):
            try:
                keys = sorted(data.keys(), key=lambda k: int(k))
            except (TypeError, ValueError):
                keys = sorted(data.keys(), key=str)
            values = [_coerce(data[k]) for k in keys]
        elif isinstance(data, (list, tuple)):
            values = []
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    values.append(_coerce(item[1]))
                else:
                    values.append(_coerce(item))
        else:
            single = _coerce(data)
            values = [single] if single is not None else None

        if values and any(v is not None for v in values):
            return values, name
    return None, None


def as_gib(value):
    """HBM metrics come back in bytes; tolerate values already in GiB."""
    if value is None:
        return None
    return value / GIB if value > 4096 else value


# --------------------------------------------------------------------------
# tpu-info gRPC backend
# --------------------------------------------------------------------------

def _attr(obj, names):
    """First present attribute from `names`, coerced to a float."""
    for name in names:
        if hasattr(obj, name):
            value = _coerce(getattr(obj, name))
            if value is not None:
                return value
    return None


class TpuInfoSource:
    """Per-chip metrics via the `tpu-info` package's gRPC client.

    jax 0.4.35 ships a libtpu with no `sdk.tpumonitoring`, but the TPU runtime
    still serves these counters on a local endpoint -- which is what the
    `tpu-info` CLI reads. Field names have moved across releases the same way
    libtpu's metric names have, so each is looked up through a candidate list.

        pip install tpu-info
    """

    label = "tpu-info gRPC"

    DUTY = ("duty_cycle_pct", "duty_cycle_percent", "duty_cycle")
    USED = ("memory_usage", "hbm_usage", "hbm_memory_usage")
    TOTAL = ("total_memory", "hbm_total", "total_hbm", "hbm_memory_total")
    TC = ("tensorcore_utilization", "tensorcore_util", "tensorcore_util_pct")

    def __init__(self):
        from tpu_info import device, metrics  # noqa: F401
        self._device = device
        self._metrics = metrics
        self._chip_type = None

    def _chip_type_once(self):
        if self._chip_type is None:
            got = self._device.get_local_chips()
            # (ChipType, count) on current releases, bare type on older ones.
            self._chip_type = got[0] if isinstance(got, tuple) else got
            if self._chip_type is None:
                raise RuntimeError("no local TPU chips reported")
        return self._chip_type

    def fetch(self):
        try:
            usages = self._metrics.get_chip_usage(self._chip_type_once())
        except Exception as exc:
            # The endpoint is only served while a workload holds the devices, so
            # a connection error here is the ordinary idle state, not a fault.
            return [], f"no chip metrics yet [{type(exc).__name__}]"
        if not usages:
            return [], "no chip metrics yet"

        ordered = sorted(usages, key=lambda u: _attr(u, ("device_id", "id")) or 0)
        chips = []
        for usage in ordered:
            chips.append({
                "duty": _attr(usage, self.DUTY),
                "hbm_used_gib": as_gib(_attr(usage, self.USED)),
                "hbm_total_gib": as_gib(_attr(usage, self.TOTAL)),
                "tc_util": _attr(usage, self.TC),
                "collective_us": None,
                "transfer_us": None,
            })
        return chips, None


def collect_chips(mon, available):
    """Return (chips, resolved_names). chips is one dict per chip on this host."""
    duty, duty_name = read_metric(mon, DUTY_CYCLE_NAMES, available)
    used, used_name = read_metric(mon, HBM_USED_NAMES, available)
    total, total_name = read_metric(mon, HBM_TOTAL_NAMES, available)
    tc, tc_name = read_metric(mon, TENSORCORE_NAMES, available)
    coll, coll_name = read_metric(mon, COLLECTIVE_NAMES, available)
    xfer, xfer_name = read_metric(mon, TRANSFER_NAMES, available)

    series = (duty, used, total, tc, coll, xfer)
    n = max((len(x) for x in series if x), default=0)

    def at(seq, i):
        return seq[i] if seq and i < len(seq) else None

    chips = []
    for i in range(n):
        chips.append({
            "duty": at(duty, i),
            "hbm_used_gib": as_gib(at(used, i)),
            "hbm_total_gib": as_gib(at(total, i)),
            "tc_util": at(tc, i),
            "collective_us": at(coll, i),
            "transfer_us": at(xfer, i),
        })
    resolved = {
        "duty": duty_name, "hbm_used": used_name,
        "hbm_total": total_name, "tc_util": tc_name,
        "collective": coll_name, "transfer": xfer_name,
    }
    return chips, resolved


def read_train_sidecar(path=TRAIN_SIDECAR, max_age=120.0):
    """Training-loop metrics, if a seqax run with slicemon_hook is live here."""
    try:
        with open(path) as fh:
            payload = json.load(fh)
    except Exception:
        return None
    ts = payload.get("ts")
    if not isinstance(ts, (int, float)) or time.time() - ts > max_age:
        return None
    return payload


# --------------------------------------------------------------------------
# host identity
# --------------------------------------------------------------------------

def metadata(path, timeout=2.0):
    req = urllib.request.Request(f"{METADATA_ROOT}/{path}", headers=METADATA_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8").strip()
    except Exception:
        return None


def detect_worker_id():
    """Worker index within the slice, per TPU VM metadata."""
    for key in ("attributes/agent-worker-number", "attributes/tpu-env"):
        raw = metadata(key)
        if not raw:
            continue
        if key.endswith("agent-worker-number"):
            try:
                return int(raw)
            except ValueError:
                continue
        for line in raw.splitlines():
            if line.startswith("WORKER_ID"):
                digits = "".join(c for c in line.split(":", 1)[-1] if c.isdigit())
                if digits:
                    return int(digits)
    # Fall back to a trailing digit group in the hostname (t1v-n-xxxx-w-3).
    host = socket.gethostname()
    parts = [p for p in host.replace("-", " ").split() if p.isdigit()]
    return int(parts[-1]) if parts else 0


# --------------------------------------------------------------------------
# publishing
# --------------------------------------------------------------------------

class LocalSink:
    def __init__(self, directory):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def put(self, name, payload):
        path = os.path.join(self.directory, name)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(payload)
        os.replace(tmp, path)

    def __str__(self):
        return self.directory


class GcsSink:
    def __init__(self, bucket, prefix):
        self.bucket_name = bucket
        self.prefix = prefix.strip("/")
        self._blob = None
        try:
            from google.cloud import storage  # type: ignore
            self._bucket = storage.Client().bucket(bucket)
            self._mode = "client"
        except Exception as exc:
            sys.stderr.write(
                f"[agent] google-cloud-storage unavailable ({exc}); "
                "falling back to the gcloud CLI\n")
            self._bucket = None
            self._mode = "cli"

    def uri(self, name):
        return f"gs://{self.bucket_name}/{self.prefix}/{name}"

    def put(self, name, payload):
        if self._mode == "client":
            blob = self._bucket.blob(f"{self.prefix}/{name}")
            blob.cache_control = "no-store"
            blob.upload_from_string(payload, content_type="application/json")
            return
        subprocess.run(
            ["gcloud", "storage", "cp", "-", self.uri(name)],
            input=payload.encode("utf-8"),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def __str__(self):
        return f"gs://{self.bucket_name}/{self.prefix}"


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    dest = ap.add_mutually_exclusive_group(required=True)
    dest.add_argument("--bucket", help="GCS bucket to publish into")
    dest.add_argument("--dir", help="local directory to publish into (testing)")
    ap.add_argument("--slice", default="slice0",
                    help="logical slice name; must match slice_monitor.py")
    ap.add_argument("--prefix", default="slicemon", help="GCS key prefix")
    ap.add_argument("--interval", type=float, default=15.0,
                    help="publish interval in seconds (default 15)")
    ap.add_argument("--worker", type=int, default=None,
                    help="override the auto-detected worker index")
    ap.add_argument("--list-metrics", action="store_true",
                    help="print the metric names libtpu supports here, then exit")
    args = ap.parse_args()

    mon = import_tpumonitoring()

    if args.list_metrics:
        if mon is None:
            print("libtpu.sdk.tpumonitoring is not importable on this host.")
            return 1
        for name in sorted(supported_metrics(mon)):
            print(name)
        return 0

    worker = args.worker if args.worker is not None else detect_worker_id()
    hostname = socket.gethostname()
    sink = LocalSink(args.dir) if args.dir else GcsSink(args.bucket, args.prefix)
    key = f"{args.slice}/host-{worker:03d}.json"
    if isinstance(sink, LocalSink):
        os.makedirs(os.path.join(args.dir, args.slice), exist_ok=True)

    available = supported_metrics(mon) if mon else set()

    # libtpu's monitoring library is absent on older builds (jax 0.4.35 has no
    # sdk submodule at all). Fall back to the tpu-info gRPC endpoint, which
    # serves the same counters.
    tpu_info = None
    if mon is None or not available:
        try:
            tpu_info = TpuInfoSource()
            backend = TpuInfoSource.label
        except Exception as exc:
            backend = f"none ({type(exc).__name__}: {exc})"
            sys.stderr.write(f"[agent] WARNING: no metric backend -- {backend}\n")
    else:
        backend = "libtpu in-process"
    print(f"[agent] metric backend: {backend}", flush=True)

    print(f"[agent] worker={worker} host={hostname} slice={args.slice} "
          f"-> {sink}/{key} every {args.interval:g}s", flush=True)
    if available:
        print(f"[agent] {len(available)} metrics available from libtpu", flush=True)

    failures = 0
    while True:
        started = time.time()
        chips, resolved, note = [], {}, None
        if tpu_info is not None:
            try:
                chips, note = tpu_info.fetch()
            except Exception as exc:
                note = f"tpu-info read failed: {type(exc).__name__}: {exc}"
        elif mon is not None:
            try:
                chips, resolved = collect_chips(mon, available)
                if not chips:
                    note = "no chip metrics yet (is a workload running?)"
            except Exception as exc:
                note = f"libtpu read failed: {type(exc).__name__}: {exc}"
        else:
            note = f"no metric backend ({backend})"

        payload = {
            "agent_version": AGENT_VERSION,
            "slice": args.slice,
            "worker": worker,
            "hostname": hostname,
            "ts": started,
            "interval": args.interval,
            "chips": chips,
            "metric_names": resolved,
            "backend": backend,
        }
        train = read_train_sidecar()
        if train is not None:
            payload["train"] = train
        if note:
            payload["note"] = note

        try:
            sink.put(key, json.dumps(payload, separators=(",", ":")))
            failures = 0
        except Exception as exc:
            failures += 1
            sys.stderr.write(f"[agent] publish failed ({failures}): {exc}\n")
            if failures >= 20:
                sys.stderr.write("[agent] giving up after 20 consecutive failures\n")
                return 1

        time.sleep(max(0.5, args.interval - (time.time() - started)))


if __name__ == "__main__":
    sys.exit(main() or 0)
