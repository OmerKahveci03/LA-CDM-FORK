"""95% confidence intervals, paired significance tests and AUROC for test-set runs.

Reads ``test_episodes.csv`` of the zero-shot run and of every ``test_step_*`` folder of
one arm, then prints:

1. Per run: point estimates with 95% intervals (patient bootstrap; Wilson for accuracy)
   for accuracy, class-mean accuracy (the paper's "mean accuracy"), macro F1, per-disease
   recall / precision / F1, tests and cost per case, ECE and Brier score of the Hypothesis
   Agent, and two AUROCs (below).
2. Paired tests of the focus step against every other run on the same patients:
   exact McNemar for accuracy, paired permutation tests for class-mean accuracy, macro
   F1, tests and cost, with paired-bootstrap intervals of the difference. Holm-adjusted
   p-values are given across the comparisons of each metric.
3. Whether the paper's LA-CDM numbers fall inside the focus step's intervals. The paper's
   per-case results are not available and its test split differs, so no p-value against
   the paper is possible.

AUROCs (the logs keep one hypothesis and a 0-10 confidence per HA call, not a probability
for every disease):

- ``auroc_correct``: how well the final HA confidence separates episodes whose final
  hypothesis is right from those where it is wrong (failure detection). Exact.
- ``auroc_ovr_approx``: one-vs-rest macro AUROC over the 4 diseases, with scores built
  from the final hypothesis and its confidence (c/10 for the stated disease, the rest
  split evenly). Approximate; a proper one needs the model's probability for every
  disease, which needs a GPU scoring pass.

Prints the report and writes it to ``<run-root>/test_step_<focus>/test_stats.md`` with the
numbers in ``test_stats.json`` beside it (``--out-dir`` changes the folder).

Standard library only.

    python scripts/test_stats.py --run-root outputs/cost_sweep/cost_1_attempt2 --focus 600
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.episode_log import DISEASES, read_rows

PAPER = {"class_mean_accuracy": 0.813, "macro_f1": 0.813, "cost": 1295.61}
PAIRED = ("accuracy", "class_mean_accuracy", "macro_f1", "tests", "cost")


# --------------------------------------------------------------------------- metrics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def auroc(scores: list[float], positives: list[bool]) -> float | None:
    """Mann-Whitney AUROC with ties counted as half."""
    pos = [s for s, y in zip(scores, positives) if y]
    neg = [s for s, y in zip(scores, positives) if not y]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=scores.__getitem__)
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, positives) if y)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def disease_scores(row: dict) -> dict[str, float]:
    """Pseudo-probabilities from the final hypothesis and its 0-10 confidence."""
    hypothesis, confidence = row["final_hypothesis"], row["final_confidence"]
    if hypothesis not in DISEASES or confidence is None:
        return {d: 1 / len(DISEASES) for d in DISEASES}
    p = min(max(confidence / 10, 0.0), 1.0)
    rest = (1 - p) / (len(DISEASES) - 1)
    return {d: p if d == hypothesis else rest for d in DISEASES}


def ha_calls(row: dict) -> list[tuple[float, bool]]:
    """(confidence 0-1, correct) for every scored HA call, as in episode_log's ECE."""
    return [(c / 10, h == row["label"]) for h, c in zip(row["hypotheses"], row["confidences"])
            if c is not None and h not in ("", "unknown") and 0 <= c <= 10]


def metrics(rows: list[dict]) -> dict[str, float | None]:
    n = len(rows)
    out: dict[str, float | None] = {
        "accuracy": sum(r["correct"] for r in rows) / n,
        "undiagnosed": sum(not r["diagnosis"] for r in rows) / n,
        "tests": sum(r["num_tests"] for r in rows) / n,
        "cost": sum(r["cost_usd"] for r in rows) / n,
    }
    recalls, f1s = [], []
    for disease in DISEASES:
        true = sum(r["label"] == disease for r in rows)
        predicted = sum(r["diagnosis"] == disease for r in rows)
        hits = sum(r["label"] == disease and r["diagnosis"] == disease for r in rows)
        recall = hits / true if true else None
        precision = hits / predicted if predicted else (0.0 if true else None)
        f1 = (2 * precision * recall / (precision + recall) if precision and recall else 0.0) if true else None
        out[f"recall_{disease}"], out[f"precision_{disease}"], out[f"f1_{disease}"] = recall, precision, f1
        if true:
            recalls.append(recall)
            f1s.append(f1)
    out["class_mean_accuracy"] = sum(recalls) / len(recalls) if recalls else None
    out["macro_f1"] = sum(f1s) / len(f1s) if f1s else None

    calls = [call for r in rows for call in ha_calls(r)]
    bins: dict[float, list[bool]] = defaultdict(list)
    for confidence, correct in calls:
        bins[confidence].append(correct)
    out["ece"] = (sum(len(v) / len(calls) * abs(c - sum(v) / len(v)) for c, v in bins.items())
                  if calls else None)
    out["brier"] = sum((c - y) ** 2 for c, y in calls) / len(calls) if calls else None

    scored = [r for r in rows if r["final_confidence"] is not None and r["final_hypothesis"] in DISEASES]
    out["auroc_correct"] = auroc([r["final_confidence"] for r in scored],
                                 [r["final_hypothesis"] == r["label"] for r in scored])
    per_class = [auroc([disease_scores(r)[d] for r in rows], [r["label"] == d for r in rows]) for d in DISEASES]
    per_class = [a for a in per_class if a is not None]
    out["auroc_ovr_approx"] = sum(per_class) / len(per_class) if per_class else None
    return out


# --------------------------------------------------------------------------- inference


def quantile(sorted_values: list[float], q: float) -> float:
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def bootstrap_intervals(rows: list[dict], resamples: int, rng: random.Random) -> dict[str, tuple]:
    point = metrics(rows)
    samples: dict[str, list[float]] = defaultdict(list)
    n = len(rows)
    for _ in range(resamples):
        for key, value in metrics([rows[rng.randrange(n)] for _ in range(n)]).items():
            if value is not None:
                samples[key].append(value)
    out = {}
    for key, value in point.items():
        values = sorted(samples[key])
        out[key] = (value, quantile(values, 0.025), quantile(values, 0.975)) if values else (value, None, None)
    k = sum(r["correct"] for r in rows)
    out["accuracy"] = (point["accuracy"], *wilson(k, n))
    return out


def mcnemar_exact(a_correct: list[bool], b_correct: list[bool]) -> tuple[int, int, float]:
    """Discordant counts (A right & B wrong, A wrong & B right) and the two-sided exact p."""
    b = sum(x and not y for x, y in zip(a_correct, b_correct))
    c = sum(y and not x for x, y in zip(a_correct, b_correct))
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, 2 * tail)


def paired(a: list[dict], b: list[dict], resamples: int, rng: random.Random) -> dict[str, dict]:
    """A minus B on the same patients: permutation p-value and bootstrap 95% interval."""
    observed_a, observed_b = metrics(a), metrics(b)
    observed = {k: observed_a[k] - observed_b[k] for k in PAIRED}
    exceed = dict.fromkeys(PAIRED, 0)
    boot: dict[str, list[float]] = defaultdict(list)
    n = len(a)
    for _ in range(resamples):
        swap = [rng.random() < 0.5 for _ in range(n)]
        pa = metrics([y if s else x for x, y, s in zip(a, b, swap)])
        pb = metrics([x if s else y for x, y, s in zip(a, b, swap)])
        for k in PAIRED:
            if abs(pa[k] - pb[k]) >= abs(observed[k]) - 1e-12:
                exceed[k] += 1
        index = [rng.randrange(n) for _ in range(n)]
        ba, bb = metrics([a[i] for i in index]), metrics([b[i] for i in index])
        for k in PAIRED:
            boot[k].append(ba[k] - bb[k])
    out = {}
    for k in PAIRED:
        values = sorted(boot[k])
        out[k] = {"diff": observed[k], "low": quantile(values, 0.025), "high": quantile(values, 0.975),
                  "p_permutation": (exceed[k] + 1) / (resamples + 1)}
    wins, losses, p = mcnemar_exact([r["correct"] for r in a], [r["correct"] for r in b])
    out["accuracy"].update({"focus_only_correct": wins, "other_only_correct": losses, "p_mcnemar": p})
    return out


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


# --------------------------------------------------------------------------- report


def load_runs(run_root: Path, zero_shot: Path | None) -> dict[str, list[dict]]:
    runs = {}
    if zero_shot and (zero_shot / "test_episodes.csv").is_file():
        runs["zero-shot"] = read_rows(zero_shot / "test_episodes.csv")
    for folder in sorted(run_root.glob("test_step_*")):
        if (folder / "test_episodes.csv").is_file():
            step = int(re.search(r"test_step_(\d+)", folder.name).group(1))
            runs[f"step {step}"] = read_rows(folder / "test_episodes.csv")
    return runs


def align(a: list[dict], b: list[dict]) -> tuple[list[dict], list[dict]]:
    """Rows of both runs for the patients they share, in the same order (first episode each)."""
    first_a = {r["patient_id"]: r for r in reversed(a)}
    first_b = {r["patient_id"]: r for r in reversed(b)}
    shared = sorted(first_a.keys() & first_b.keys())
    return [first_a[p] for p in shared], [first_b[p] for p in shared]


def fmt(key: str, value) -> str:
    if value is None:
        return "-"
    if key == "cost":
        return f"${value:,.0f}"
    if key == "tests":
        return f"{value:.2f}"
    if key.startswith("auroc") or key == "brier":
        return f"{value:.3f}"
    return f"{100 * value:.1f}%"


def fmt_diff(key: str, value: float) -> str:
    if key == "cost":
        return f"{value:+,.0f} $"
    if key == "tests":
        return f"{value:+.2f}"
    return f"{100 * value:+.1f} pts"


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


ROWS = (["accuracy", "class_mean_accuracy", "macro_f1", "undiagnosed", "tests", "cost", "ece", "brier",
         "auroc_correct", "auroc_ovr_approx"]
        + [f"{m}_{d}" for d in DISEASES for m in ("recall", "precision", "f1")])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, default=Path("outputs/cost_sweep/cost_1"),
                        help="Arm folder holding test_step_* (default: outputs/cost_sweep/cost_1).")
    parser.add_argument("--zero-shot", type=Path, default=Path("outputs/zero_shot/test"))
    parser.add_argument("--focus", type=int, default=600, help="Step compared against every other run.")
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path,
                        help="Where test_stats.md and test_stats.json go (default: <run-root>/test_step_<focus>/).")
    args = parser.parse_args()
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    runs = load_runs(args.run_root, args.zero_shot)
    focus = f"step {args.focus}"
    if focus not in runs:
        sys.exit(f"No {args.run_root}/test_step_{args.focus:04d}/test_episodes.csv (found: {', '.join(runs) or 'none'})")
    rng = random.Random(args.seed)

    emit(f"# Test-set statistics ({args.resamples} resamples, seed {args.seed})\n")
    intervals = {name: bootstrap_intervals(rows, args.resamples, rng) for name, rows in runs.items()}
    names = list(runs)
    emit("## 95% intervals per run\n")
    emit("| Metric | " + " | ".join(f"{n} (n={len(runs[n])})" for n in names) + " |")
    emit("|---|" + "---|" * len(names))
    for key in ROWS:
        cells = []
        for name in names:
            value, low, high = intervals[name][key]
            cells.append(fmt(key, value) + ("" if low is None else f" [{fmt(key, low)}, {fmt(key, high)}]"))
        emit(f"| {key} | " + " | ".join(cells) + " |")

    comparisons = {}
    for name in names:
        if name != focus:
            a, b = align(runs[focus], runs[name])
            comparisons[name] = paired(a, b, args.resamples, rng) | {"n_shared": len(a)}
    if comparisons:
        for key in PAIRED:
            raw = [c[key]["p_mcnemar" if key == "accuracy" else "p_permutation"] for c in comparisons.values()]
            for c, adjusted in zip(comparisons.values(), holm(raw)):
                c[key]["p_holm"] = adjusted
        emit(f"\n## {focus} minus each other run (same patients)\n")
        emit("Accuracy p: exact McNemar; others: paired permutation. Holm across the comparisons per metric.\n")
        emit("| vs | Metric | Difference [95% CI] | p | p (Holm) |")
        emit("|---|---|---|---|---|")
        for name, c in comparisons.items():
            for key in PAIRED:
                d = c[key]
                p = d["p_mcnemar" if key == "accuracy" else "p_permutation"]
                extra = (f" ({d['focus_only_correct']} vs {d['other_only_correct']} discordant)"
                         if key == "accuracy" else "")
                emit(f"| {name} (n={c['n_shared']}) | {key} | {fmt_diff(key, d['diff'])} "
                      f"[{fmt_diff(key, d['low'])}, {fmt_diff(key, d['high'])}]{extra} | {fmt_p(p)} | {fmt_p(d['p_holm'])} |")

    emit(f"\n## {focus} vs the paper's LA-CDM (cost reward)\n")
    emit("| Metric | Paper | Ours [95% CI] | Paper inside our CI? |")
    emit("|---|---|---|---|")
    for key, paper in PAPER.items():
        value, low, high = intervals[focus][key]
        emit(f"| {key} | {fmt(key, paper)} | {fmt(key, value)} [{fmt(key, low)}, {fmt(key, high)}] | "
              f"{'yes' if low <= paper <= high else 'no'} |")

    out_dir = args.out_dir or args.run_root / f"test_step_{args.focus:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "test_stats.json").write_text(json.dumps(
        {"focus": focus, "intervals": intervals, "comparisons": comparisons, "paper": PAPER,
         "resamples": args.resamples, "seed": args.seed}, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'test_stats.md'} and {out_dir / 'test_stats.json'}")


if __name__ == "__main__":
    main()
