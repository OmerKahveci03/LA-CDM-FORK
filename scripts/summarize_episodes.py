"""Print Markdown tables from episode CSVs written by training or evaluation.

Evaluation files (``test_episodes.csv``, ``val_step_*_episodes.csv``) give
overall and per-disease results, end reasons, a confusion matrix, and how
often each test was ordered. A training file (``train_episodes.csv``) gives
per-block learning-signal statistics; ``cost_only_signal`` is the fraction of
steps whose four rollouts tied on diagnosis and format, so their advantages
came only from test cost.

    python scripts/summarize_episodes.py outputs/cost_sweep/cost_1/test_step_0300/test_episodes.csv
    python scripts/summarize_episodes.py outputs/cost_sweep/cost_1/train/episodes/train_episodes.csv --block 25
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.episode_log import DISEASES, END_REASONS, UNDIAGNOSED, group_signal_flags, read_rows, summarize


def fmt(value, percent: bool = False) -> str:
    if value is None:
        return "-"
    if percent:
        return f"{100 * value:.1f}%"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def evaluation_report(rows: list[dict]) -> str:
    summary = summarize(rows)
    groups = [("all", summary["overall"])] + list(summary["per_disease"].items())
    out = ["### Overall and per disease", table(
        ["Group", "n", "Accuracy", "Undiagnosed", "Acc. (diagnosed)", "Precision", "F1",
         "Tests/case", "Cost/case", "Turns", "HA acc."],
        [
            [
                name,
                fmt(g["n"]),
                fmt(g["accuracy"], True),
                fmt(g["undiagnosed_fraction"], True),
                fmt(g["accuracy_diagnosed"], True),
                fmt(g.get("precision"), True),
                fmt(g.get("f1", g.get("macro_f1")), True),
                fmt(g["avg_num_tests"]),
                "$" + fmt(g["avg_cost_usd"]) if g["avg_cost_usd"] is not None else "-",
                fmt(g["avg_turns"]),
                fmt(g["hypothesis_accuracy"], True),
            ]
            for name, g in groups
        ],
    )]
    out.append("\nThe `all` row's F1 is macro F1. HA acc. is over every Hypothesis Agent call.\n")
    out.append("### How episodes ended")
    out.append(table(["Group"] + list(END_REASONS), [
        [name] + [fmt(g["end_reasons"][r], True) for r in END_REASONS] for name, g in groups
    ]))
    out.append("\n### Confusion matrix (rows: true disease)")
    predictions = list(DISEASES) + [UNDIAGNOSED]
    out.append(table(["True \\ predicted"] + predictions, [
        [label] + [str(summary["confusion"][label][p]) for p in predictions] for label in DISEASES
    ]))
    tests = sorted(summary["overall"]["test_rates"], key=lambda t: -summary["overall"]["test_rates"][t])
    out.append("\n### Share of episodes in which each test returned a result")
    out.append(table(["Test"] + [name for name, _ in groups], [
        [test] + [fmt(g["test_rates"].get(test, 0.0), True) for _, g in groups] for test in tests
    ]))
    return "\n".join(out)


def training_report(rows: list[dict], block: int) -> str:
    by_step = defaultdict(list)
    for row in rows:
        by_step[row["step"]].append(row)
    blocks = defaultdict(list)
    for step, step_rows in sorted(by_step.items()):
        blocks[(step - 1) // block].append((step, step_rows))
    lines = []
    for index, entries in sorted(blocks.items()):
        flags = [group_signal_flags(step_rows) for _, step_rows in entries]
        flat = [r for _, step_rows in entries for r in step_rows]

        def mean(key, source):
            values = [s[key] for s in source if s.get(key) is not None]
            return sum(values) / len(values) if values else None

        lines.append([
            f"{entries[0][0]}-{entries[-1][0]}",
            entries[0][1][0]["phase"],
            fmt(mean("correct_fraction", flags), True),
            fmt(mean("invalid_fraction", flags), True),
            fmt(mean("cost_only_signal", flags), True),
            fmt(mean("no_signal", flags), True),
            fmt(mean("reward_total", flat)),
            fmt(mean("reward_test_cost", flat)),
            fmt(mean("num_tests", flat)),
        ])
    return table(
        ["Steps", "Phase", "Correct", "Undiagnosed", "Cost-only signal", "No signal",
         "Reward", "Cost reward", "Tests/episode"],
        lines,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--block", type=int, default=25, help="Steps per row for training files.")
    args = parser.parse_args()
    for path in args.paths:
        rows = read_rows(path)
        print(f"## {path} ({len(rows)} episodes)\n")
        if rows and rows[0]["stage"] == "train":
            print(training_report(rows, args.block))
        else:
            print(evaluation_report(rows))
        print()


if __name__ == "__main__":
    main()
