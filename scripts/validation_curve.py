"""Collect every validation pass of a training run into one table over steps.

Reads ``val_step_<N>_summary.json`` files written during training, prints a
Markdown table, and writes ``val_curve.csv`` beside them. Each step is one
training patient, so the step is also the number of patients trained on.

    python scripts/validation_curve.py outputs/cost_sweep/cost_1/train

Validation is sampled (four rollouts per case at the training temperature),
so differences of a few points between neighbouring steps can be noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.episode_log import DISEASES

CYCLE_STEPS = 300
PHASE_ENDS = {100: "after DA", 200: "after HA-SFT", 0: "after calibration"}


def stage_note(step: int) -> str:
    if step == 0:
        return "untrained"
    note = PHASE_ENDS.get(step % CYCLE_STEPS, "")
    cycle = (step - 1) // CYCLE_STEPS + 1
    return f"cycle {cycle} {note}".strip() if note else f"cycle {cycle}"


def load_curve(train_dir: Path) -> list[dict]:
    episodes = train_dir / "episodes" if (train_dir / "episodes").is_dir() else train_dir
    points = []
    for path in sorted(episodes.glob("val_step_*_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        overall = summary["overall"]
        point = {
            "step": summary["step"],
            "stage": stage_note(summary["step"]),
            "accuracy": overall["accuracy"],
            "undiagnosed": overall["undiagnosed_fraction"],
            "accuracy_diagnosed": overall["accuracy_diagnosed"],
            "macro_f1": overall.get("macro_f1"),
            "tests_per_case": overall["avg_num_tests"],
            "turns": overall.get("avg_turns"),
            "cost_usd": overall["avg_cost_usd"],
            "hypothesis_accuracy": overall["hypothesis_accuracy"],
            "ece": overall.get("ece"),
            "invalid_format": overall["end_reasons"].get("invalid_format"),
            "max_steps": overall["end_reasons"].get("max_steps"),
            "max_length": overall["end_reasons"].get("max_length"),
        }
        for disease in DISEASES:
            point[f"accuracy_{disease}"] = summary["per_disease"].get(disease, {}).get("accuracy")
        points.append(point)
    return sorted(points, key=lambda p: p["step"])


def cell(key: str, value) -> str:
    if value is None:
        return "-"
    if key in ("step", "stage"):
        return str(value)
    if key == "cost_usd":
        return f"${value:,.0f}"
    if key in ("tests_per_case",):
        return f"{value:.2f}"
    if key == "ece":
        return f"{100 * value:.1f}"
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train_dir", type=Path, nargs="?", default=Path("outputs/cost_sweep/cost_1/train"),
                        help="Run's train/ directory or its episodes/ directory "
                             "(default: outputs/cost_sweep/cost_1/train).")
    args = parser.parse_args()
    points = load_curve(args.train_dir)
    if not points:
        sys.exit(f"No val_step_*_summary.json files under {args.train_dir}")

    shown = ["step", "stage", "accuracy", "undiagnosed", "accuracy_diagnosed", "macro_f1",
             "tests_per_case", "cost_usd", "hypothesis_accuracy", "ece"]
    print("| " + " | ".join(shown) + " |")
    print("|" + "|".join("---" for _ in shown) + "|")
    for point in points:
        print("| " + " | ".join(cell(k, point[k]) for k in shown) + " |")

    per_disease = ["step"] + [f"accuracy_{d}" for d in DISEASES]
    print("\n| " + " | ".join(per_disease) + " |")
    print("|" + "|".join("---" for _ in per_disease) + "|")
    for point in points:
        print("| " + " | ".join(cell(k, point[k]) for k in per_disease) + " |")

    episodes = args.train_dir / "episodes" if (args.train_dir / "episodes").is_dir() else args.train_dir
    out = episodes / "val_curve.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
