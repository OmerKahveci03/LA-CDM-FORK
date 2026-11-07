"""Summarize training reward logs by the three 100-step loss phases."""

import argparse
import ast
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    rows = []
    for line in args.log.open(encoding="utf-8", errors="replace"):
        if line.startswith("{'loss':"):
            rows.append(ast.literal_eval(line))
    print(f"Logged training steps: {len(rows)}")
    for name, start, end in (("DA", 0, 100), ("HA SFT", 100, 200), ("HA calibration", 200, 300)):
        phase = rows[start:end]
        if not phase:
            continue
        lengths = [row["completion_length"] for row in phase if "completion_length" in row]
        length_text = f"{mean(lengths):.1f}" if lengths else "unavailable"
        print(
            f"{name} ({start + 1}-{start + len(phase)}): "
            f"format-positive steps={sum(row['rewards/FormatReward'] > 0 for row in phase)}/{len(phase)}, "
            f"diagnosis-positive steps={sum(row['rewards/DiagnosisReward'] > 0 for row in phase)}/{len(phase)}, "
            f"mean cost reward={mean(row['rewards/TestCostReward'] for row in phase):.4f}, "
            f"mean completion length={length_text}"
        )


if __name__ == "__main__":
    main()
