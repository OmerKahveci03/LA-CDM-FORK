"""Hypothesis Agent calibration at each validation step, split by call type.

The HA is called once from the patient history alone, then again after every
test result. As the Decision Agent orders fewer tests, more of its calls are
history-only. This table shows, for each kind of call, how confident the HA was
(mean confidence / 10) versus how often it was right, so a rising ECE can be
traced to a shift in the mix of calls or to calibration getting worse.

    python scripts/calibration_by_call.py          # default: outputs/cost_sweep/cost_1/train
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.episode_log import read_rows

DEFAULT_RUN = Path("outputs/cost_sweep/cost_1/train")


def cell(calls: list[tuple[float, bool]]) -> str:
    if not calls:
        return f"{'-':>6} {'-':>6} {'-':>6} {'-':>6}"
    n = len(calls)
    confidence = sum(c for c, _ in calls) / n
    accuracy = sum(a for _, a in calls) / n
    return f"{n:6d} {confidence:6.2f} {accuracy:6.2f} {confidence - accuracy:+6.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train_dir", type=Path, nargs="?", default=DEFAULT_RUN)
    args = parser.parse_args()
    episodes = args.train_dir / "episodes" if (args.train_dir / "episodes").is_dir() else args.train_dir
    paths = sorted(episodes.glob("val_step_*_episodes.csv"))
    if not paths:
        sys.exit(f"No val_step_*_episodes.csv under {episodes}")

    print("conf = mean stated confidence / 10, acc = share of those calls naming the true disease, "
          "gap = conf - acc (positive = overconfident)\n")
    print(f"{'':>5} | {'history-only (1st call)':^27} | {'after a test result':^27} | {'all calls':^27}")
    print(f"{'step':>5} | " + " | ".join([f"{'n':>6} {'conf':>6} {'acc':>6} {'gap':>6}"] * 3))
    for path in paths:
        step = int(re.search(r"val_step_(\d+)_", path.name).group(1))
        first, later = [], []
        for row in read_rows(path):
            for index, (hypothesis, confidence) in enumerate(zip(row["hypotheses"], row["confidences"])):
                if confidence is None or hypothesis in ("", "unknown"):
                    continue
                (first if index == 0 else later).append((confidence / 10, hypothesis == row["label"]))
        print(f"{step:5d} | {cell(first)} | {cell(later)} | {cell(first + later)}")


if __name__ == "__main__":
    main()
