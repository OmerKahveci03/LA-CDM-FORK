"""Compare one-rollout test metrics with the reported trained Llama reference.

Reference values come from weekly_reports/week_5/WEEKLY_REPORT.tex. Pass a
test metrics.json from src.evaluate, not four-rollout training validation.
"""

import argparse
import json
from pathlib import Path


METRICS = (
    ("Overall accuracy", "eval_all/accuracy", 0.6875, "percent"),
    ("Diagnosed-only accuracy", "eval_ignore/accuracy", 0.7021, "percent"),
    ("Undiagnosed fraction", "eval_none_fraction", 0.0208, "percent"),
    ("Action-format reward", "eval_rewards/FormatReward", 0.9792, "decimal"),
    ("Macro F1", "eval_all/macro_f1", 0.6627, "percent"),
    ("Hypothesis accuracy", "eval_hypothesis_accuracy", 0.6326, "percent"),
    ("Expected calibration error", "eval_ece", 0.1038, "percent"),
    ("Average tests", "eval_avg_num_tests", 2.73, "number"),
    ("Average diagnostic cost", "eval_avg_diagnostic_cost", 1137.67, "dollars"),
)


def display(value, kind):
    if value is None:
        return "missing"
    if kind == "percent":
        return f"{value * 100:.2f}%"
    if kind == "dollars":
        return f"${value:,.2f}"
    return f"{value:.4f}" if kind == "decimal" else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_file", type=Path, help="Test metrics.json from src.evaluate")
    args = parser.parse_args()
    with args.metrics_file.open(encoding="utf-8") as file:
        candidate = json.load(file)
    if not isinstance(candidate, dict):
        parser.error("metrics file must contain a JSON object")

    print(f"Candidate: {args.metrics_file}")
    print("Reference: original trained Llama, one rollout on 240 test cases (Week 5 report)")
    print("Metric | Original | Candidate | Change")
    print("--- | ---: | ---: | ---:")
    for label, key, baseline, kind in METRICS:
        value = candidate.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            value = None
        difference = value - baseline if value is not None else None
        if difference is None:
            change = "missing"
        elif kind == "percent":
            change = f"{difference * 100:+.2f} pp"
        elif kind == "dollars":
            change = f"${difference:+,.2f}"
        else:
            change = f"{difference:+.2f}"
        print(f"{label} | {display(baseline, kind)} | {display(value, kind)} | {change}")


if __name__ == "__main__":
    main()
