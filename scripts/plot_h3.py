"""Figures for H3 (docs/HYPOTHESIS_H3.md): the DA retrained against the frozen separately fine-tuned HA.

    python scripts/plot_h3.py                       # defaults below; writes figures/h3/

Writes .png and .pdf (plus the plotted values as ``*_points.csv``):

h3_validation         validation curves of H3a and H3b: accuracy, undiagnosed share, tests and
                      cost per case. The chosen checkpoint (step 700) is marked. There are no
                      phase labels because every H3 step trains the DA.
h3_test_overview      the four test runs side by side: the DA's diagnosis, the HA's last
                      hypothesis, tests and cost per case, with 95% intervals
h3_test_rates         share of test patients given each test, per run
h3_test_confusion     one test confusion matrix per run
h3_cost_vs_accuracy   each run as a point, cost against accuracy

Styling, metrics and intervals come from ``scripts/plot_run.py``, so these figures match the
Week 7 ones. Run on HiPerGator, where the outputs live.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_run import (  # noqa: E402
    AXIS, GRID, INK_2, MATRIX_NOTE, MUTED, PREDICTED, SERIES, SURFACE,
    base_style, category_axis, draw_ci_bars, draw_matrix, load_validation, new_figure, save,
    test_metrics, write_points, y_format,
)
from src.utils.episode_log import DISEASES, read_rows  # noqa: E402

R1 = "outputs/cost_sweep/cost_1_attempt2"
TEST_RUNS = [  # (label, test folder); fixed order, so each run keeps its color in every figure
    ("Baseline model", f"{R1}/test_step_0600"),
    ("Baseline DA + separately fine-tuned HA", f"{R1}/frozen_ha_seed0_step_0600"),
    ("Retrained DA (H3a)", "outputs/h3/h3a_ha0/frozen_ha_seed0_step_0700"),
    ("Retrained DA + belief reward (H3b)", "outputs/h3/h3b_ha0/frozen_ha_seed0_step_0700"),
]
TRAIN_RUNS = [("H3a", "outputs/h3/h3a_ha0/train"), ("H3b", "outputs/h3/h3b_ha0/train")]
CHOSEN_STEP = 700
VALIDATION_PANELS = [
    ("accuracy", "Accuracy (undiagnosed counted wrong)", "pct"),
    ("undiagnosed", "Undiagnosed share", "pct"),
    ("tests_per_case", "Average tests per case", "number"),
    ("cost_usd", "Average test cost per case (USD)", "money"),
]


def short(label: str) -> str:
    """Two-line tick label."""
    return label.replace(" + ", "\n+ ").replace("separately ", "separately\n").replace(" (", "\n(")


def plot_validation(out_dir: Path) -> None:
    from matplotlib.ticker import MultipleLocator

    runs = {name: load_validation(Path(path), 100) for name, path in TRAIN_RUNS}
    runs = {name: pts for name, pts in runs.items() if pts}
    colors = {name: SERIES[2 + i] for i, (name, _) in enumerate(TRAIN_RUNS)}  # same hues as the test figures
    max_step = max(p["step"] for pts in runs.values() for p in pts)
    fig, axes = new_figure(2, 2, 4.2, 3.0)
    for ax, (key, title, kind) in zip(axes.flat, VALIDATION_PANELS):
        base_style(ax, title)
        ax.set_xlim(-max_step * 0.02, max_step * 1.08)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.axvline(CHOSEN_STEP, color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.text(CHOSEN_STEP, 1.0, "chosen checkpoint", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color=MUTED)
        top = 0.0
        for name, pts in runs.items():
            series = [(p["step"], p[key]) for p in pts if p.get(key) is not None]
            if not series:
                continue
            xs, ys = zip(*series)
            ax.plot(xs, ys, color=colors[name], linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, clip_on=False, zorder=3)
            ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
            top = max(top, max(ys))
        y_format(ax, kind, 1 if kind == "pct" else top)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(runs), frameon=False, fontsize=8, labelcolor=INK_2)
    save(fig, out_dir / "h3_validation", title="H3 validation (240 cases x 4 sampled rollouts, every 100 steps)",
         note="Every step trains the Decision Agent; the separately fine-tuned HA is frozen. H3a was stopped by the collapse "
              "guard at step 900.")
    write_points(out_dir / "h3_validation_points.csv",
                 [{"run": name, "step": p["step"], **{k: p.get(k) for k, _, _ in VALIDATION_PANELS}}
                  for name, pts in runs.items() for p in pts])


def with_final_hypothesis(rows: list[dict]) -> list[dict]:
    """The same rows, scored by the HA's last hypothesis instead of the DA's diagnosis."""
    out = []
    for r in rows:
        guess = (r.get("final_hypothesis") or "").lower()
        out.append({**r, "diagnosis": guess if guess in DISEASES else "", "correct": guess == r["label"]})
    return out


def plot_test(out_dir: Path, resamples: int) -> None:
    runs = [(name, read_rows(Path(path) / "test_episodes.csv")) for name, path in TEST_RUNS
            if (Path(path) / "test_episodes.csv").is_file()]
    missing = [name for name, path in TEST_RUNS if not (Path(path) / "test_episodes.csv").is_file()]
    if missing:
        print(f"  missing test runs: {', '.join(missing)}")
    if not runs:
        return
    names = [name for name, _ in runs]
    palette = {name: SERIES[i] for i, (name, _) in enumerate(TEST_RUNS)}
    colors = [palette[n] for n in names]
    ticks = [short(n) for n in names]
    da = {name: test_metrics(rows, resamples) for name, rows in runs}
    ha = {name: test_metrics(with_final_hypothesis(rows), resamples) for name, rows in runs}

    # 1. Overview.
    panels = [
        (da, "accuracy", "Accuracy: the DA's diagnosis", "pct"),
        (ha, "accuracy", "Accuracy: the HA's last hypothesis", "pct"),
        (da, "tests", "Average tests per case", "number"),
        (da, "cost", "Average test cost per case (USD)", "money"),
    ]
    fig, axes = new_figure(2, 2, 5.2, 3.4)
    for ax, (source, key, title, kind) in zip(axes.flat, panels):
        category_axis(ax, title, ticks)
        draw_ci_bars(ax, ticks, [source[n][key] for n in names], colors, kind)
    save(fig, out_dir / "h3_test_overview", title=f"Test set ({len(runs[0][1])} cases, 1 rollout each)",
         note="Error bars: 95% intervals (Wilson for accuracy; case bootstrap otherwise). "
              "Paired differences are in docs/HYPOTHESIS_H3.md.")
    write_points(out_dir / "h3_test_points.csv",
                 [{"run": n, "score": score, "metric": k, "value": v[0], "ci_low": v[1], "ci_high": v[2]}
                  for score, source in (("diagnosis", da), ("final_hypothesis", ha))
                  for n in names for k, v in source[n].items() if isinstance(v, tuple)])

    # 2. Which tests each run orders.
    rates = {name: {} for name in names}
    for name, rows in runs:
        for r in rows:
            for test in {t.lower() for t in r["tests"]}:
                rates[name][test] = rates[name].get(test, 0) + 1 / len(rows)
    tests = sorted({t for r in rates.values() for t, v in r.items() if v >= 0.01},
                   key=lambda t: -max(rates[n].get(t, 0) for n in names))
    fig, axes = new_figure(1, 1, 10, 3.8)
    ax = axes[0][0]
    base_style(ax, "Share of test patients given each test")
    ax.set_xlabel("")
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        xs = [j + (i - (len(names) - 1) / 2) * width for j in range(len(tests))]
        ax.bar(xs, [rates[name].get(t, 0) for t in tests], width=width * 0.92, color=palette[name],
               edgecolor=SURFACE, linewidth=1.0, label=name, zorder=3)
    ax.set_xticks(range(len(tests)), [t.replace(" ", "\n", 1) for t in tests], fontsize=8, color=INK_2)
    y_format(ax, "pct", 1)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK_2)
    save(fig, out_dir / "h3_test_rates", note="Tests ordered by fewer than 1% of patients in every run are left out.")
    write_points(out_dir / "h3_test_rates_points.csv",
                 [{"run": n, "test": t, "share": round(rates[n].get(t, 0), 4)} for n in names for t in tests])

    # 3. Confusion matrices.
    fig, axes = new_figure(1, len(runs), 3.2, 3.4)
    for ax, (name, rows) in zip(axes.flat, runs):
        matrix = []
        for true in DISEASES:
            group = [r for r in rows if r["label"] == true]
            matrix.append([sum((r["diagnosis"] or "undiagnosed") == pred for r in group) / len(group)
                           if group else 0 for pred in PREDICTED])
        draw_matrix(ax, matrix, short(name).replace("\n", " ", 1))
    axes[0][0].set_ylabel("True disease", fontsize=8, color=INK_2)
    save(fig, out_dir / "h3_test_confusion", title="Test-set confusion matrices (the DA's diagnosis)",
         note=MATRIX_NOTE)

    # 4. Cost against accuracy.
    from matplotlib.ticker import FuncFormatter

    fig, axes = new_figure(1, 1, 6, 4.2)
    ax = axes[0][0]
    base_style(ax, "Accuracy (the DA's diagnosis) against test cost")
    ax.set_xlabel("Average test cost per case (USD)", fontsize=8, color=INK_2)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for name in names:
        cost, acc = da[name]["cost"], da[name]["accuracy"]
        ax.errorbar(cost[0], acc[0], xerr=[[cost[0] - cost[1]], [cost[2] - cost[0]]],
                    yerr=[[acc[0] - acc[1]], [acc[2] - acc[0]]], fmt="o", color=palette[name],
                    ecolor=palette[name], elinewidth=1.0, capsize=2, markersize=8, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=4, label=name)
    ax.set_ylim(0.6, 1.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlim(0, max(da[n]["cost"][2] for n in names) * 1.15)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_2)
    save(fig, out_dir / "h3_cost_vs_accuracy", note="Bars: 95% intervals. Up and to the left is better. "
         "The y-axis starts at 60%.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("figures/h3"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_validation(args.out_dir)
    plot_test(args.out_dir, args.bootstrap)
    print(f"Wrote to {args.out_dir}/: h3_validation, h3_test_overview, h3_test_rates, h3_test_confusion, "
          "h3_cost_vs_accuracy")


if __name__ == "__main__":
    main()
