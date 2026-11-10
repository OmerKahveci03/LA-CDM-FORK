"""Draw every figure for a training run with one command.

    python scripts/plot_run.py                      # run: outputs/cost_sweep/cost_1/train
    python scripts/plot_run.py --compare attempt1=outputs/cost_sweep/cost_1_attempt1_collapsed/train

Writes into ``--out-dir`` (default ``figures/``), each as .png and .pdf with the
plotted values in a ``_points.csv`` beside it:

validation_overview       accuracy, undiagnosed share, test cost per case
validation_per_disease    accuracy for each disease
validation_details        diagnosed-only accuracy, macro F1, HA accuracy, tests and
                          turns per case, ECE, and how episodes ended
validation_distributions  per validation step: tests and turns per episode (stacked
                          shares), test cost and transcript length (box plots)
training                  per-step training rollouts (``train_episodes.csv``),
                          averaged over ``--window`` steps, incl. the GRPO signal
confusion_over_time       per true disease, what the model predicted at each step
confusion_matrices        full matrices at step 0, every cycle end, and the last step

and the same per disease in ``per_disease/<disease>/`` (validation_details with
recall, precision, F1 and the rest; validation_distributions; training), unless
``--no-per-disease``.

Validation figures read ``val_step_<N>_summary.json`` / ``val_step_<N>_episodes.csv``
(every 100 steps, the same 240 patients x 4 rollouts each time); use them to judge
the model and choose checkpoints. The training figure reads the rollouts the model
trained on (one new patient per step); use it to see why validation moved.

``--compare LABEL=PATH`` (repeatable) adds other runs to the three validation-curve
figures. PATH is a run's ``train/`` directory or, for runs before November 2026, its
Slurm training log (validated every ``--log-eval-steps`` steps; no test cost). The
untrained model is the same for every run, so a step-0 point measured in one run is
shown for runs that lack one (``--no-shared-step0`` disables this).
``--only`` draws a subset of figures.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validation_curve import load_curve
from src.utils.episode_log import DISEASES, group_signal_flags, read_rows

DEFAULT_RUN = Path("outputs/cost_sweep/cost_1/train")
FIGURES = ("validation", "distributions", "training", "confusion")

# Reference palette (light mode). Categorical hues are assigned to runs in order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Ordinal blue ramp (steps 250-650), validated with validate_palette.js --mode light --ordinal.
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
BOX_FILL, BOX_LINE = "#cde2fb", "#1c5cab"
# Predicted class in confusion charts: the first 4 categorical hues (validated
# together: CVD worst adjacent dE 9.1) in DISEASES order; undiagnosed is neutral gray.
DISEASE_COLORS = dict(zip(DISEASES, SERIES[:4]))
UNDIAGNOSED_COLOR = AXIS
# Sequential blue ramp (steps 100-700) for confusion-matrix heatmaps.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Mean vs median on the test-count and cost panels: mean keeps the run's color,
# the median takes the next categorical hue.
MEDIAN_COLOR = SERIES[1]
MEDIAN_OF = {"tests_per_case": "tests_median", "cost_usd": "cost_median",
             "tests": "tests_median", "cost": "cost_median"}
CYCLE, DA_STEPS, PHASES = 300, 100, ("DA", "HA-SFT", "Calib.")
X_LABEL = "Patients trained on (steps)"

# Validation-curve panels: (key, title, kind). kind: "pct" (fixed 0-100%),
# "pct_auto" (percent, range from data), "money", "number" (range from data).
OVERVIEW = [
    ("accuracy", "Accuracy (undiagnosed counted wrong)", "pct"),
    ("undiagnosed", "Undiagnosed share", "pct"),
    ("cost_usd", "Average test cost per case (USD)", "money"),
]
DETAILS = [
    ("accuracy_diagnosed", "Accuracy among diagnosed episodes", "pct"),
    ("macro_f1", "Macro F1 (diseases weighted equally)", "pct"),
    ("hypothesis_accuracy", "Hypothesis Agent accuracy (every call)", "pct"),
    ("tests_per_case", "Average tests per case", "number"),
    ("turns", "Decision Agent turns per case", "number"),
    ("ece", "Calibration error of HA confidence (ECE)", "pct_auto"),
    ("invalid_format", "Ended: unparseable action", "pct_auto"),
    ("max_length", "Ended: hit the token limit", "pct_auto"),
    ("max_steps", "Ended: hit the turn limit", "pct_auto"),
]
# One disease's validation curve (from the per-disease block of each summary).
DISEASE_DETAILS = [
    ("accuracy", "Accuracy / recall (undiagnosed counted wrong)", "pct"),
    ("undiagnosed", "Undiagnosed share", "pct"),
    ("accuracy_diagnosed", "Accuracy among diagnosed episodes", "pct"),
    ("precision", "Precision (predictions of it that were right)", "pct"),
    ("f1", "F1", "pct"),
    ("hypothesis_accuracy", "Hypothesis Agent accuracy (every call)", "pct"),
    ("tests_per_case", "Average tests per case", "number"),
    ("cost_usd", "Average test cost per case (USD)", "money"),
    ("turns", "Decision Agent turns per case", "number"),
    ("ece", "Calibration error of HA confidence (ECE)", "pct_auto"),
    ("invalid_format", "Ended: unparseable action", "pct_auto"),
    ("max_length", "Ended: hit the token limit", "pct_auto"),
]
# Distribution panels. Small integers as stacked shares: (column, title, labels, value -> index).
SHARE_PANELS = [
    ("num_tests", "Tests per episode (share of episodes)", ["0", "1", "2", "3", "4+"], lambda v: min(v, 4)),
    ("num_turns", "Decision Agent turns per episode", ["1", "2", "3", "4", "5+"], lambda v: min(max(v, 1), 5) - 1),
]
BOX_PANELS = [
    ("cost_usd", "Test cost per episode (USD), spread", "money"),
    ("transcript_tokens", "Transcript length (tokens)", "number"),
]
TRAINING_PANELS = [
    ("correct", "Rollouts diagnosed correctly", "pct"),
    ("undiagnosed", "Rollouts undiagnosed", "pct"),
    ("tests", "Average tests per rollout", "number"),
    ("cost", "Average test cost per rollout (USD)", "money"),
    ("max_length", "Rollouts that hit the token limit", "pct_auto"),
    ("signal", "Patients the model learned from", "pct"),
    ("cost_only", "Patients where only test cost differed", "pct_auto"),
]


# ---------------------------------------------------------------- shared styling

def episodes_dir(train_dir: Path) -> Path:
    return train_dir / "episodes" if (train_dir / "episodes").is_dir() else train_dir


def base_style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=12)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(X_LABEL, fontsize=8, color=INK_2)


def y_format(ax, kind: str, top: float) -> None:
    from matplotlib.ticker import FuncFormatter

    ax.set_ylim(0, 1 if kind == "pct" else (top * 1.15 if top > 0 else 1))
    if kind in ("pct", "pct_auto"):
        # Small shares (e.g. 0.1% turn-limit endings) need decimals or every tick reads "0%".
        decimals = 0 if kind == "pct" or top >= 0.05 else (1 if top >= 0.005 else 2)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.{decimals}f}%"))
    elif kind == "money":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.2f}" if top < 10 else f"${v:,.0f}"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.2f}" if top < 10 else f"{v:,.0f}"))


def new_figure(rows: int, columns: int, width: float, height: float):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def save(fig, stem: Path, note: str | None = None, title: str | None = None) -> None:
    import matplotlib.pyplot as plt

    top = 1.0
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
        top = 0.96
    bottom = 0.0
    if note:
        fig.text(0.01, 0.005, note, fontsize=8, color=MUTED)
        bottom = 0.03
    fig.tight_layout(rect=(0, bottom, 1, top))
    for suffix in (".pdf", ".png"):
        fig.savefig(stem.with_name(stem.name + suffix), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def write_points(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------ validation curves

def points_from_log(path: Path, eval_steps: int) -> list[dict]:
    """Parse the validation dictionaries an older training job printed, in order."""
    points = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        start = line.find("{'eval_")
        if start < 0 or "eval_all/accuracy" not in line:
            continue
        try:
            metrics = ast.literal_eval(line[start : line.rindex("}") + 1])
        except (ValueError, SyntaxError):
            continue
        point = {
            "step": eval_steps * (len(points) + 1),
            "accuracy": metrics.get("eval_all/accuracy"),
            "undiagnosed": metrics.get("eval_none_fraction"),
            "cost_usd": metrics.get("eval_avg_diagnostic_cost"),
        }
        for disease in DISEASES:
            point[f"accuracy_{disease}"] = metrics.get(f"eval_all/recall_{disease}")
        points.append(point)
    return points


def load_validation(path: Path, eval_steps: int) -> list[dict]:
    return load_curve(path) if path.is_dir() else points_from_log(path, eval_steps)


def load_summaries(train_dir: Path) -> list[dict]:
    summaries = [json.loads(p.read_text(encoding="utf-8"))
                 for p in episodes_dir(train_dir).glob("val_step_*_summary.json")]
    return sorted(summaries, key=lambda s: s["step"])


def median(values: list[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def add_medians(points: list[dict], episodes: dict[int, list[dict]], disease: str | None = None) -> None:
    """Add per-step medians of tests and cost from the validation episode CSVs
    (the summary JSONs only store means)."""
    for point in points:
        rows = episodes.get(point["step"], [])
        if disease:
            rows = [r for r in rows if r["label"] == disease]
        if rows:
            point["tests_median"] = median([r["num_tests"] for r in rows])
            point["cost_median"] = median([r["cost_usd"] for r in rows])


def draw_median(ax, points: list[dict], key: str, mean_color: str) -> float:
    """Draw the median beside the mean on a tests/cost panel; returns its max."""
    from matplotlib.lines import Line2D

    series = [(p["step"], p[MEDIAN_OF[key]]) for p in points if p.get(MEDIAN_OF[key]) is not None]
    if not series:
        return 0.0
    xs, ys = zip(*series)
    ax.plot(xs, ys, color=MEDIAN_COLOR, linewidth=1.5, solid_capstyle="round", solid_joinstyle="round",
            clip_on=False, zorder=3)
    ax.legend(handles=[Line2D([], [], color=mean_color, linewidth=1.5, label="mean"),
                       Line2D([], [], color=MEDIAN_COLOR, linewidth=1.5, label="median")],
              loc="upper right", frameon=False, fontsize=8, labelcolor=INK_2, handlelength=1.5)
    return max(ys)


def load_disease_curve(summaries: list[dict], disease: str) -> list[dict]:
    points = []
    for summary in summaries:
        group = summary.get("per_disease", {}).get(disease)
        if not group:
            continue
        ends = group.get("end_reasons", {})
        points.append({
            "step": summary["step"], "accuracy": group.get("accuracy"),
            "undiagnosed": group.get("undiagnosed_fraction"), "accuracy_diagnosed": group.get("accuracy_diagnosed"),
            "precision": group.get("precision"), "f1": group.get("f1"),
            "hypothesis_accuracy": group.get("hypothesis_accuracy"), "tests_per_case": group.get("avg_num_tests"),
            "cost_usd": group.get("avg_cost_usd"), "turns": group.get("avg_turns"), "ece": group.get("ece"),
            "invalid_format": ends.get("invalid_format"), "max_length": ends.get("max_length"),
        })
    return points


def plot_disease_validation(summaries: list[dict], episodes: dict[int, list[dict]], disease: str, label: str,
                            out_dir: Path) -> list[str]:
    points = load_disease_curve(summaries, disease)
    if not points:
        return []
    add_medians(points, episodes, disease)
    max_step = max(p["step"] for p in points)
    curve_figure({label: points}, {label: SERIES[0]}, DISEASE_DETAILS, out_dir / "validation_details", 3, max_step,
                 title=f"Validation: {disease} (true disease)")
    write_points(out_dir / "validation_points.csv", points)
    return ["validation_details"]


def curve_axis(ax, title: str, kind: str, max_step: int) -> None:
    from matplotlib.ticker import MultipleLocator

    base_style(ax, title)
    if kind == "pct":
        y_format(ax, kind, 1)
    ax.set_xlim(-max_step * 0.02, max_step * 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(100 if max_step <= 900 else CYCLE))
    # Phase boundaries: every 100 steps on short runs, cycle ends only on long ones.
    for step in range(100, max_step + 1, 100 if max_step <= 900 else CYCLE):
        ax.axvline(step, color=GRID if step % CYCLE else AXIS, linewidth=0.8, zorder=0)
    if max_step <= 900:
        for start in range(0, max_step, 100):
            ax.text(start + 50, 1.0, PHASES[(start // 100) % 3], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7, color=MUTED)


def curve_figure(runs: dict, colors: dict, panels: list, stem: Path, columns: int, max_step: int,
                 title: str | None = None) -> None:
    rows = (len(panels) + columns - 1) // columns
    fig, axes = new_figure(rows, columns, 3.6, 2.9)
    for ax, (key, panel_title, kind) in zip(axes.flat, panels):
        curve_axis(ax, panel_title, kind, max_step)
        drawn = False
        for name, points in runs.items():
            series = [(p["step"], p[key]) for p in points if p.get(key) is not None]
            if not series:
                continue
            xs, ys = zip(*series)
            dense = len(series) > 10  # smaller markers so ring gaps don't make the line look dashed
            ax.plot(xs, ys, color=colors[name], linewidth=1.5, solid_capstyle="round", solid_joinstyle="round",
                    marker="o", markersize=3.5 if dense else 5, markeredgecolor=SURFACE,
                    markeredgewidth=0.8 if dense else 1.5, label=name, clip_on=False, zorder=3)
            if 2 <= len(runs) <= 4 and len(series) >= 2:  # one run: the title names it
                ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK_2)
            drawn = True
        median_top = 0.0
        if drawn and len(runs) == 1 and key in MEDIAN_OF:  # comparisons stay mean-only
            name, points = next(iter(runs.items()))
            median_top = draw_median(ax, points, key, colors[name])
        if not drawn:
            ax.text(0.5, 0.5, "not recorded for these runs", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color=MUTED)
            ax.set_yticks([])
        elif kind != "pct":
            top = max(p[key] for pts in runs.values() for p in pts if p.get(key) is not None)
            y_format(ax, kind, max(top, median_top))
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    if len(runs) >= 2:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=len(runs), frameon=False, fontsize=8, labelcolor=INK_2)
        save(fig, stem, title=title or " ")
    else:
        save(fig, stem, title=title)


def plot_validation(runs: dict, out_dir: Path, shared_step0: bool) -> list[str]:
    if shared_step0:
        step0 = next((p for points in runs.values() for p in points if p["step"] == 0), None)
        if step0 is not None:
            for name, points in runs.items():
                if not any(p["step"] == 0 for p in points):
                    points.insert(0, step0)
                    print(f"  {name}: using the shared step-0 (untrained) validation point")
    colors = {name: SERIES[i] for i, name in enumerate(runs)}
    max_step = max(p["step"] for points in runs.values() for p in points)
    per_disease = [(f"accuracy_{d}", f"{d.capitalize()}: accuracy", "pct") for d in DISEASES]
    curve_figure(runs, colors, OVERVIEW, out_dir / "validation_overview", 3, max_step)
    curve_figure(runs, colors, per_disease, out_dir / "validation_per_disease", 2, max_step)
    curve_figure(runs, colors, DETAILS, out_dir / "validation_details", 3, max_step)
    keys = (["step"] + [k for k, _, _ in OVERVIEW + DETAILS] + ["tests_median", "cost_median"]
            + [k for k, _, _ in per_disease])
    write_points(out_dir / "validation_points.csv",
                 [{"run": name, **{k: p.get(k) for k in keys}} for name, points in runs.items() for p in points])
    return ["validation_overview", "validation_per_disease", "validation_details"]


# ------------------------------------------------------ validation distributions

def load_validation_episodes(train_dir: Path, disease: str | None) -> dict[int, list[dict]]:
    steps = {}
    for path in sorted(episodes_dir(train_dir).glob("val_step_*_episodes.csv")):
        rows = read_rows(path)
        if disease:
            rows = [r for r in rows if r["label"] == disease]
        if rows:
            steps[int(re.search(r"val_step_(\d+)_", path.name).group(1))] = rows
    return steps


def quantile(sorted_values: list[float], q: float) -> float:
    position = (len(sorted_values) - 1) * q
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def step_axis(ax, title: str, steps: list[int]) -> None:
    base_style(ax, title)
    spacing = min((b - a for a, b in zip(steps, steps[1:])), default=100)
    ax.set_xlim(steps[0] - spacing * 0.6, steps[-1] + spacing * 0.6)
    every = CYCLE if steps[-1] - steps[0] > 900 else 100
    ax.set_xticks([s for s in steps if s % every == 0])


def plot_distributions(train_dir: Path, out_dir: Path, disease: str | None, token_limit: int) -> list[str]:
    data = load_validation_episodes(train_dir, disease)
    if not data:
        print(f"  skipped validation_distributions: no val_step_*_episodes.csv under {train_dir}")
        return []
    steps = sorted(data)
    width = min((b - a for a, b in zip(steps, steps[1:])), default=100) * 0.7
    fig, axes = new_figure(2, 2, 6, 3.5)
    points = []

    for ax, (column, title, labels, to_index) in zip(axes[0], SHARE_PANELS):
        step_axis(ax, title, steps)
        y_format(ax, "pct", 1)
        shares = []
        for step in steps:
            counts = [0] * len(labels)
            for row in data[step]:
                counts[to_index(row[column])] += 1
            shares.append([c / len(data[step]) for c in counts])
        bottoms = [0.0] * len(steps)
        for index, label in enumerate(labels):
            heights = [s[index] for s in shares]
            # Surface-coloured edges give the gap between stacked segments.
            ax.bar(steps, heights, width=width, bottom=bottoms, color=ORDINAL[index], label=label,
                   edgecolor=SURFACE, linewidth=1.0, zorder=3)
            bottoms = [b + h for b, h in zip(bottoms, heights)]
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8,
                  labelcolor=INK_2, handlelength=1.0, reverse=True)
        points += [{"panel": column, "step": s, **{f"share_{l}": round(v, 4) for l, v in zip(labels, share)}}
                   for s, share in zip(steps, shares)]

    for ax, (column, title, kind) in zip(axes[1], BOX_PANELS):
        step_axis(ax, title, steps)
        stats = []
        for step in steps:
            values = sorted(float(r[column]) for r in data[step])
            stats.append({"p05": quantile(values, 0.05), "q1": quantile(values, 0.25),
                          "median": quantile(values, 0.5), "q3": quantile(values, 0.75),
                          "p95": quantile(values, 0.95), "mean": sum(values) / len(values)})
        ax.bxp([{"med": s["median"], "q1": s["q1"], "q3": s["q3"], "whislo": s["p05"], "whishi": s["p95"],
                 "mean": s["mean"], "fliers": []} for s in stats],
               positions=steps, widths=width, showmeans=True, showfliers=False, patch_artist=True,
               boxprops={"facecolor": BOX_FILL, "edgecolor": BOX_LINE, "linewidth": 1.0},
               medianprops={"color": BOX_LINE, "linewidth": 2.0},
               whiskerprops={"color": MUTED, "linewidth": 1.0}, capprops={"color": MUTED, "linewidth": 1.0},
               meanprops={"marker": "o", "markerfacecolor": INK, "markeredgecolor": SURFACE,
                          "markeredgewidth": 1.0, "markersize": 4},
               manage_ticks=False, zorder=3)
        top = max(s["p95"] for s in stats)
        if column == "transcript_tokens" and token_limit:
            ax.axhline(token_limit, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
            ax.text(steps[-1] + width * 0.6, token_limit, f"training limit {token_limit:,}",
                    ha="right", va="bottom", fontsize=7, color=MUTED)
            top = max(top, token_limit)
        y_format(ax, kind if kind == "money" else "number", top / 1.15 * 1.12 if top else 0)
        points += [{"panel": column, "step": s, **{k: round(v, 2) for k, v in st.items()}}
                   for s, st in zip(steps, stats)]

    who = f"{disease} episodes" if disease else "all episodes"
    name = "validation_distributions"
    save(fig, out_dir / name,
         note="Boxes: middle 50%, line = median, whiskers = 5th-95th percentile, dot = mean. "
              "Tests = distinct tests that returned a result (not unavailable or misnamed requests); cost is theirs.",
         title=f"Validation distributions ({who}, {len(data[steps[0]])} per step)")
    write_points(out_dir / f"{name}_points.csv", points)
    return [name]


# --------------------------------------------------------------- training rollouts

def load_training_rows(train_dir: Path) -> list[dict]:
    path = train_dir if train_dir.is_file() else episodes_dir(train_dir) / "train_episodes.csv"
    rows = read_rows(path) if path.is_file() else []
    if not rows:
        print(f"  skipped training: no rows in {path}")
    return rows


def plot_training(rows: list[dict], out_dir: Path, window: int, disease: str | None = None) -> list[str]:
    """Training rollouts averaged over windows of ``window`` steps.

    With ``disease``, only steps whose patient has that true disease are used, so
    a window holds fewer patients (about prevalence x window) and is noisier.
    """
    if disease:
        rows = [r for r in rows if r["label"] == disease]
    if not rows:
        return []
    by_step = defaultdict(list)
    for row in rows:
        by_step[row["step"]].append(row)
    per_step = {}
    for step, group in by_step.items():
        flags = group_signal_flags(group)
        n = len(group)
        per_step[step] = {
            "correct": sum(r["correct"] for r in group) / n,
            "undiagnosed": sum(not r["diagnosis"] for r in group) / n,
            "tests": sum(r["num_tests"] for r in group) / n,
            "cost": sum(r["cost_usd"] for r in group) / n,
            "max_length": sum(r["end_reason"] == "max_length" for r in group) / n,
            "signal": 1.0 - flags["no_signal"] if flags else None,
            "cost_only": flags.get("cost_only_signal") if flags else None,
        }
    binned = defaultdict(list)
    for step, metrics in per_step.items():
        binned[(step - 1) // window].append(metrics)
    rollout_values = defaultdict(lambda: {"tests": [], "cost": []})
    for row in rows:
        rollout_values[(row["step"] - 1) // window]["tests"].append(row["num_tests"])
        rollout_values[(row["step"] - 1) // window]["cost"].append(row["cost_usd"])
    last = max(per_step)
    points = []
    for index in sorted(binned):
        group = binned[index]
        point = {"step": min((index + 1) * window, last), "steps_in_window": len(group)}
        for key, _, _ in TRAINING_PANELS:
            values = [m[key] for m in group if m[key] is not None]
            point[key] = sum(values) / len(values) if values else None
        point["tests_median"] = median(rollout_values[index]["tests"])
        point["cost_median"] = median(rollout_values[index]["cost"])
        points.append(point)

    from matplotlib.ticker import MultipleLocator

    fig, axes = new_figure(2, 4, 3.6, 2.9)
    for ax in list(axes.flat)[len(TRAINING_PANELS):]:
        ax.set_visible(False)
    for ax, (key, title, kind) in zip(axes.flat, TRAINING_PANELS):
        base_style(ax, title)
        series = [(p["step"], p[key]) for p in points if p[key] is not None]
        top = max((y for _, y in series), default=0)
        if key in MEDIAN_OF:
            top = max([top] + [p[MEDIAN_OF[key]] for p in points if p.get(MEDIAN_OF[key]) is not None])
        y_format(ax, kind, top)
        ax.set_xlim(0, last)
        ax.xaxis.set_major_locator(MultipleLocator(CYCLE))
        for start in range(0, last, CYCLE):  # shade Decision Agent phases
            ax.axvspan(start, min(start + DA_STEPS, last), color=GRID, alpha=0.45, linewidth=0, zorder=0)
        if series:
            xs, ys = zip(*series)
            ax.plot(xs, ys, color=SERIES[0], linewidth=1.5, solid_capstyle="round", solid_joinstyle="round", zorder=3)
            if key in MEDIAN_OF:
                draw_median(ax, points, key, SERIES[0])
    if disease:
        title = f"Training rollouts over time: {disease} patients only"
        span = f"Each point covers {window} steps (only those with a {disease} patient; see _points.csv)."
    else:
        title = "Training rollouts over time"
        span = f"Each point averages {window} steps ({4 * window} rollouts)."
    save(fig, out_dir / "training", title=title,
         note=f"{span} Shaded: Decision Agent phases. Tests = distinct tests that returned a result. "
              "Each step: 4 rollouts of one patient; GRPO learns only when their rewards differ.")
    write_points(out_dir / "training_points.csv", points)
    return ["training"]


# ----------------------------------------------------------- confusion over time

def plot_confusion(summaries: list[dict], out_dir: Path) -> list[str]:
    summaries = [s for s in summaries if s.get("confusion")]
    if not summaries:
        print("  skipped confusion: no confusion matrices in the summaries")
        return []
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    from matplotlib.patches import Patch

    predicted = list(DISEASES) + ["undiagnosed"]
    colors = [DISEASE_COLORS[d] for d in DISEASES] + [UNDIAGNOSED_COLOR]
    steps = [s["step"] for s in summaries]
    shares = {}  # (step, true, predicted) -> share of that true disease's episodes
    points = []
    for summary in summaries:
        for true in DISEASES:
            row = summary["confusion"].get(true, {})
            total = sum(row.values())
            for pred in predicted:
                count = row.get(pred, 0)
                shares[(summary["step"], true, pred)] = count / total if total else 0.0
                points.append({"step": summary["step"], "true": true, "predicted": pred, "count": count,
                               "share_of_true": round(count / total, 4) if total else None})
    write_points(out_dir / "confusion_points.csv", points)

    # 1. Over time: for each true disease, what the model predicted at each step.
    width = min((b - a for a, b in zip(steps, steps[1:])), default=100) * 0.7
    fig, axes = new_figure(2, 2, 6, 3.3)
    for ax, true in zip(axes.flat, DISEASES):
        step_axis(ax, f"True {true}: what the model predicted", steps)
        y_format(ax, "pct", 1)
        bottoms = [0.0] * len(steps)
        for pred, color in zip(predicted, colors):
            heights = [shares[(s, true, pred)] for s in steps]
            ax.bar(steps, heights, width=width, bottom=bottoms, color=color, edgecolor=SURFACE,
                   linewidth=1.0, zorder=3)
            bottoms = [b + h for b, h in zip(bottoms, heights)]
    handles = [Patch(color=c, label=("predicted " + p) if p != "undiagnosed" else "undiagnosed")
               for p, c in zip(predicted, colors)]
    fig.legend(handles=handles, loc="upper right", ncol=5, frameon=False, fontsize=8, labelcolor=INK_2)
    save(fig, out_dir / "confusion_over_time", title="Confusion over validation steps",
         note="Each bar is one true disease's validation episodes; the segment in that disease's own color "
              "is correct (its recall).")

    # 2. Full matrices at the step-0 point, every cycle end and the last step.
    chosen = [s for s in steps if s == 0 or s % CYCLE == 0 or s == steps[-1]]
    columns = min(4, len(chosen))
    rows = (len(chosen) + columns - 1) // columns
    fig, axes = new_figure(rows, columns, 3.4, 3.2)
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    image = None
    for ax, step in zip(axes.flat, chosen):
        matrix = [[shares[(step, t, p)] for p in predicted] for t in DISEASES]
        image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"Step {step}", loc="left", fontsize=10, color=INK, pad=8)
        ax.set_xticks(range(len(predicted)), [p[:5] + "." if p != "undiagnosed" else "none" for p in predicted],
                      fontsize=7, color=INK_2)
        ax.set_yticks(range(len(DISEASES)), [d[:5] + "." for d in DISEASES], fontsize=7, color=INK_2)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        for i, line in enumerate(matrix):
            for j, value in enumerate(line):
                r, g, b = to_rgb(cmap(value))
                light = 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5
                ax.text(j, i, f"{100 * value:.0f}", ha="center", va="center", fontsize=7,
                        color="#0b0b0b" if light else "#ffffff")
    for ax in list(axes.flat)[len(chosen):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("True disease", fontsize=8, color=INK_2)
    save(fig, out_dir / "confusion_matrices", title="Confusion matrices (% of each true disease's episodes)",
         note="Rows: true disease. Columns: prediction (appen. cholec. divert. pancr., none = undiagnosed). "
              "Rows sum to 100.")
    return ["confusion_over_time", "confusion_matrices"]


# ---------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train_dir", type=Path, nargs="?", default=DEFAULT_RUN,
                        help=f"Run's train/ directory (default: {DEFAULT_RUN}).")
    parser.add_argument("--label", default="this run", help="Name of the run in comparison legends.")
    parser.add_argument("--compare", action="append", default=[], metavar="LABEL=PATH",
                        help="Another run for the validation-curve figures (repeatable).")
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--only", choices=FIGURES, action="append", help="Draw only these figures (repeatable).")
    parser.add_argument("--window", type=int, default=25, help="Training steps per plotted point (default 25).")
    parser.add_argument("--no-per-disease", action="store_true", help="Skip the per_disease/<disease>/ figures.")
    parser.add_argument("--token-limit", type=int, default=2048, help="Line on the transcript panel (0 hides it).")
    parser.add_argument("--log-eval-steps", type=int, default=100)
    parser.add_argument("--no-shared-step0", action="store_true")
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        sys.exit("matplotlib is required: pip install --no-cache-dir matplotlib")
    if not args.train_dir.exists():
        sys.exit(f"No run at {args.train_dir}")

    selected = set(args.only or FIGURES)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # Per-episode validation rows (for medians); empty for a Slurm-log run.
    val_episodes = (load_validation_episodes(args.train_dir, None)
                    if "validation" in selected and args.train_dir.is_dir() else {})
    if "validation" in selected:
        runs = {args.label: load_validation(args.train_dir, args.log_eval_steps)}
        add_medians(runs[args.label], val_episodes)
        for item in args.compare:
            name, _, path = item.partition("=")
            if not path:
                sys.exit(f"--compare must be LABEL=PATH, got {item!r}")
            runs[name] = load_validation(Path(path), args.log_eval_steps)
        empty = [name for name, points in runs.items() if not points]
        if empty:
            print(f"  no validation results for: {', '.join(empty)}")
        runs = {name: points for name, points in runs.items() if points}
        if len(runs) > len(SERIES):
            sys.exit(f"At most {len(SERIES)} runs per chart")
        if runs:
            written += plot_validation(runs, args.out_dir, not args.no_shared_step0)
    if "distributions" in selected:
        written += plot_distributions(args.train_dir, args.out_dir, None, args.token_limit)
    training_rows = load_training_rows(args.train_dir) if "training" in selected else []
    if training_rows:
        written += plot_training(training_rows, args.out_dir, args.window)
        print(f"  training: {len(training_rows)} rollouts over {len({r['step'] for r in training_rows})} steps")
    summaries = load_summaries(args.train_dir) if args.train_dir.is_dir() else []
    if "confusion" in selected:
        written += plot_confusion(summaries, args.out_dir)
    if written:
        print(f"Wrote to {args.out_dir}/: {', '.join(written)}")

    if not args.no_per_disease:
        for disease in DISEASES:
            folder = args.out_dir / "per_disease" / disease
            folder.mkdir(parents=True, exist_ok=True)
            done = []
            if "validation" in selected:
                done += plot_disease_validation(summaries, val_episodes, disease, args.label, folder)
            if "distributions" in selected:
                done += plot_distributions(args.train_dir, folder, disease, args.token_limit)
            if training_rows:
                done += plot_training(training_rows, folder, args.window, disease)
            if done:
                print(f"Wrote to {folder}/: {', '.join(done)}")
            written += done

    if not written:
        sys.exit("Nothing to plot.")
    print("Each figure is a .png and .pdf, with its plotted values in a matching _points.csv.")


if __name__ == "__main__":
    main()
