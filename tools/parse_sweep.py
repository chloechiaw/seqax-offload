"""Turn a directory of seqax run logs into one CSV row per run.

Reads only the logs, never the TPU, so it works after the allocation is gone. Runs that
crashed or OOMed produce a row with a `status` other than `ok` rather than being dropped --
a missing point and a failed point mean different things when reading the curve.

  python3 tools/parse_sweep.py results/logs > results/v6e_sweep.csv
"""

import csv
import os
import re
import sys

PATTERNS = {
    "step_s_for_2": r"Profile time: ([0-9.]+)s for 2 steps",
    "mfu_pct": r"MFU \(projections only\): ([0-9.]+)% MFU",
    "peak_hbm_gib": r"True peak training memory per device: ([0-9.]+) GiB",
    "params": r"Model params: ([0-9_]+)",
    "tokens_per_step": r"^Tokens: ([0-9_]+)",
    "device_kind": r"Device kind: (.+)",
    "dcn_total_mib": r"^\s+total\s+([0-9.]+) MiB",
    "dcn_predicted_ms": r"predicted DCN time at [0-9.]+ GB/s: ([0-9]+) ms",
}

FIELDS = [
    "run", "arm", "batch", "device_kind", "chips", "tokens_per_chip",
    "step_ms", "mfu_pct", "peak_hbm_gib", "dcn_total_mib", "dcn_predicted_ms", "status",
]


def parse(path):
    text = open(path, errors="replace").read()
    row = {"run": os.path.basename(path).replace(".log", "")}

    m = re.match(r"arm(\d)_b(\d+)", row["run"])
    row["arm"], row["batch"] = (m.group(1), int(m.group(2))) if m else ("", "")

    for key, pat in PATTERNS.items():
        found = re.search(pat, text, re.MULTILINE)
        row[key] = found.group(1).replace("_", "") if found else ""

    # arm 0 is one slice of 8 chips; arm 1 is two.
    row["chips"] = 8 if row["arm"] == "0" else 16 if row["arm"] == "1" else ""
    if row["step_s_for_2"]:
        row["step_ms"] = round(float(row["step_s_for_2"]) / 2 * 1000, 1)
    if row["tokens_per_step"] and row["chips"]:
        row["tokens_per_chip"] = int(row["tokens_per_step"]) // row["chips"]

    if "RESOURCE_EXHAUSTED" in text or "Out of memory" in text:
        row["status"] = "oom"
    elif not row["step_ms"]:
        row["status"] = "failed"
    elif "halted unexpectedly" in text:
        row["status"] = "ok_then_crashed"  # profiler kills multi-host runs after the number prints
    else:
        row["status"] = "ok"
    return row


def main(logdir):
    rows = [parse(os.path.join(logdir, f)) for f in sorted(os.listdir(logdir)) if f.endswith(".log")]
    rows.sort(key=lambda r: (str(r["arm"]), r["batch"] if isinstance(r["batch"], int) else 0))
    w = csv.DictWriter(sys.stdout, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/logs")
