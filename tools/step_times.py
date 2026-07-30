"""Step time from the per-step log timestamps, not the built-in 2-step profile.

train.py's "Profile time: Xs for 2 steps" is a two-sample estimate taken on steps 1-2, which
also happen to be the steps the profiler is attached to. Measured against itself across
identical configs it varies by ~30%, which is larger than most effects worth measuring.

training_io.log stamps every step, so N-1 intervals are already recorded. Timestamps are
whole seconds, so a single interval is coarse -- but total elapsed divided by interval count
converges quickly, and by step 18 it is good to a few tens of ms.

  python3 tools/step_times.py results/logs/*.log
"""

import datetime
import re
import sys

STEP = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[.\d]*\] Step (\d+):", re.MULTILINE)


def step_ms(path, warmup=3):
    """Mean ms/step over the steady-state window, and how many intervals it averages."""
    hits = [
        (datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S"), int(n))
        for t, n in STEP.findall(open(path, errors="replace").read())
    ]
    # Drop warmup: step 0 includes compilation, and steps 1-2 carry the profiler.
    hits = [(t, n) for t, n in hits if n >= warmup]
    if len(hits) < 3:
        return None, 0
    span = (hits[-1][0] - hits[0][0]).total_seconds()
    intervals = hits[-1][1] - hits[0][1]
    if intervals <= 0 or span <= 0:
        return None, 0
    return span / intervals * 1000, intervals


def profile_ms(path):
    m = re.search(r"Profile time: ([0-9.]+)s for 2 steps", open(path, errors="replace").read())
    return float(m.group(1)) / 2 * 1000 if m else None


if __name__ == "__main__":
    print(f"{'run':<26} {'steady ms':>10} {'n':>4} {'2-step ms':>10} {'delta':>8}")
    for path in sys.argv[1:]:
        steady, n = step_ms(path)
        prof = profile_ms(path)
        name = path.split("/")[-1].replace(".log", "")
        if steady is None:
            print(f"{name:<26} {'-':>10} {0:>4} {(f'{prof:.0f}' if prof else '-'):>10}")
            continue
        # prof is absent whenever SEQAX_PROFILE is unset, which is the normal case for
        # multi-host runs -- the built-in profile block is what kills them.
        delta = f"{(prof - steady) / steady * 100:+.0f}%" if prof else "-"
        print(f"{name:<26} {steady:>10.0f} {n:>4} {(f'{prof:.0f}' if prof else '-'):>10} {delta:>8}")
