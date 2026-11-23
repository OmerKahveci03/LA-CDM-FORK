"""Figures for H4 (docs/HYPOTHESIS_H4.md): the contrastive (CLM-style) hypothesis agent.

    python scripts/plot_h4.py                                   # writes figures/h4/
    python scripts/plot_h4.py --joint-step 600 --frozen-step 400

Writes .png and .pdf (plus the plotted values as ``*_points.csv``):

h4_heads              stage 1: the heads' accuracy per epoch for both backbones (chosen epoch
                      marked), and the joint arm's online heads loss at every DA step
h4_ha_test            H4a: each HA alone on the same four test record sets, with 95% intervals
h4_validation         stage 4: validation curves of the joint and frozen arms (DA accuracy,
                      undiagnosed share, HA accuracy, HA calibration, tests and cost per case),
                      each arm's chosen step marked; whatever steps exist are plotted, so this
                      also works while the 1,900-step extension is still running
h4_test_overview      H4b and H4c test runs side by side with the reference systems
h4_test_rates         share of test patients given each test, per run
h4_test_confusion     one test confusion matrix per run
h4_cost_vs_accuracy   each run as a point, cost against accuracy

Missing inputs are skipped with a message. Styling, metrics and intervals come from
``scripts/plot_run.py`` so the figures match the H3 and Week 7 ones. Run on HiPerGator (CPU job):

    sbatch --export=ALL,SCRIPT=jobs/figures_h3_h4.sh slurm/cpu_script.sbatch
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_run import (  # noqa: E402
    AXIS, GRID, INK_2, MATRIX_NOTE, MUTED, PREDICTED, SERIES, SURFACE,
    base_style, category_axis, draw_ci_bars, draw_matrix, load_validation, new_figure, save,
    test_metrics, wilson, write_points, y_format,
)
from src.utils.episode_log import DISEASES, read_rows  # noqa: E402

R1 = "outputs/cost_sweep/cost_1_attempt2"
CON = "outputs/contrastive"
TRAIN_RUNS = [("Heads learning (joint)", f"{CON}/joint/joint_h2/train"),
              ("Heads frozen", f"{CON}/joint/frozen_h2/train")]
PREREGISTERED_STEPS = 700
VALIDATION_PANELS = [
    ("accuracy", "DA accuracy (undiagnosed counted wrong)", "pct"),
    ("undiagnosed", "Undiagnosed share", "pct"),
    ("hypothesis_accuracy", "HA accuracy (all hypotheses)", "pct"),
    ("ece", "HA calibration error (ECE)", "pct_auto"),
    ("tests_per_case", "Average tests per case", "number"),
    ("cost_usd", "Average test cost per case (USD)", "money"),
]
HA_RECORD_SETS = [("A_da2", "Baseline DA's tests"), ("B_none", "No tests"),
                  ("C_random2", "2 random tests"), ("D_all", "All tests")]
HA_RUNS = [  # (label, folder holding <record set>/test_episodes.csv)
    ("CLM HA (h2 backbone)", f"{CON}/scores/h2"),
    ("CLM HA (plain Llama)", f"{CON}/scores/base"),
    ("New HA (H2)", "outputs/ha_sft/scores/new0"),
    ("Baseline HA", "outputs/ha_sft/scores/old2"),
]


def test_runs(joint_step: int, frozen_step: int) -> list[tuple[str, str]]:
    """Fixed order, so each run keeps its colour in every test figure."""
    return [
        ("Baseline model", f"{R1}/test_step_0600"),
        ("Baseline DA + new HA", f"{R1}/frozen_ha_seed0_step_0600"),
        ("Baseline DA + CLM HA (H4b)", f"{CON}/combined/h2"),
        ("Retrained DA + new HA (H3a)", "outputs/h3/h3a_ha0/frozen_ha_seed0_step_0700"),
        (f"New DA + CLM HA, heads learning (step {joint_step})",
         f"{CON}/joint/joint_h2/test_step_{joint_step:04d}"),
        (f"New DA + CLM HA, heads frozen (step {frozen_step})",
         f"{CON}/joint/frozen_h2/test_step_{frozen_step:04d}"),
    ]


def short(label: str) -> str:
    """Multi-line tick label."""
    return label.replace(" + ", "\n+ ").replace(", ", ",\n").replace(" (", "\n(")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ------------------------------------------------------------ stage 1 and the online heads

def plot_heads(out_dir: Path) -> None:
    curves = {b: read_csv(Path(CON) / "heads" / b / "curve.csv")
              for b in ("h2", "base") if (Path(CON) / "heads" / b / "curve.csv").is_file()}
    updates_path = Path(TRAIN_RUNS[0][1]) / "contrastive_heads_updates.jsonl"
    updates = []
    if updates_path.is_file():
        for line in updates_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                updates.append(json.loads(line))
        updates = sorted({u["step"]: u for u in updates}.values(), key=lambda u: u["step"])  # resumes repeat steps
    if not curves and not updates:
        print("  skipped h4_heads: no heads/<backbone>/curve.csv and no heads update log")
        return
    from matplotlib.lines import Line2D

    fig, axes = new_figure(1, 2, 5.2, 3.4)
    ax = axes[0][0]
    base_style(ax, "Stage 1: heads accuracy per epoch")
    ax.set_xlabel("Epoch", fontsize=8, color=INK_2)
    colors = {"h2": SERIES[0], "base": SERIES[1]}
    names = {"h2": "Llama + H2 adapter (chosen)", "base": "Plain Llama"}
    points = []
    for b, rows in curves.items():
        epochs = [int(r["epoch"]) for r in rows]
        val = [float(r["val_accuracy"]) for r in rows]
        train = [float(r["train_accuracy"]) for r in rows]
        ax.plot(epochs, val, color=colors[b], linewidth=2, zorder=3)
        ax.plot(epochs, train, color=colors[b], linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        best = max(rows, key=lambda r: (float(r["val_accuracy"]), -float(r["val_loss"])))
        ax.scatter([int(best["epoch"])], [float(best["val_accuracy"])], s=60, color=colors[b],
                   edgecolors=SURFACE, linewidths=1.5, zorder=4)
        ax.annotate(f"epoch {best['epoch']}: {100 * float(best['val_accuracy']):.1f}%",
                    (int(best["epoch"]), float(best["val_accuracy"])), xytext=(4, 9),
                    textcoords="offset points", fontsize=7, color=INK_2)
        points += [{"figure": "stage1", "backbone": b, **r} for r in rows]
    y_format(ax, "pct", 1)
    handles = [Line2D([], [], color=colors[b], linewidth=2, label=names[b]) for b in curves]
    handles += [Line2D([], [], color=MUTED, linewidth=2, label="validation"),
                Line2D([], [], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), label="training")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7, labelcolor=INK_2)

    ax = axes[0][1]
    base_style(ax, "Joint arm: heads loss at each online update")
    if updates:
        steps = [u["step"] for u in updates]
        losses = [u["loss_first"] for u in updates]
        window = 25
        smooth = [sum(losses[max(0, i - window + 1): i + 1]) / len(losses[max(0, i - window + 1): i + 1])
                  for i in range(len(losses))]
        ax.plot(steps, losses, color=SERIES[0], linewidth=0.6, alpha=0.35, zorder=2)
        ax.plot(steps, smooth, color=SERIES[0], linewidth=2, zorder=3, label=f"mean of last {window} steps")
        if max(steps) > PREREGISTERED_STEPS:
            ax.axvline(PREREGISTERED_STEPS, color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        y_format(ax, "number", max(losses))
        ax.legend(loc="upper right", frameon=False, fontsize=7, labelcolor=INK_2)
        points += [{"figure": "online", "step": u["step"], "loss": u["loss_first"],
                    "online_rows": u.get("online_rows"), "rows": u.get("rows")} for u in updates]
    save(fig, out_dir / "h4_heads",
         note="Left: the chosen epoch maximises validation accuracy. Right: loss on that step's new HA "
              "records plus 32 replay rows, before the update (cross-entropy over the 4 diseases).")
    write_points(out_dir / "h4_heads_points.csv", points)


# ------------------------------------------------------------ H4a: the HAs alone

def plot_ha_test(out_dir: Path) -> None:
    available = [(name, Path(folder)) for name, folder in HA_RUNS
                 if any((Path(folder) / s / "test_episodes.csv").is_file() for s, _ in HA_RECORD_SETS)]
    if not available:
        print("  skipped h4_ha_test: no HA score folders")
        return
    fig, axes = new_figure(1, 1, 10, 3.8)
    ax = axes[0][0]
    base_style(ax, "Each HA alone: accuracy of its hypothesis on the same test records")
    ax.set_xlabel("")
    width = 0.8 / len(available)
    points = []
    for i, (name, folder) in enumerate(available):
        color = SERIES[[n for n, _ in HA_RUNS].index(name)]
        for j, (record_set, _) in enumerate(HA_RECORD_SETS):
            path = folder / record_set / "test_episodes.csv"
            if not path.is_file():
                continue
            rows = read_rows(path)
            k, n = sum(r["correct"] for r in rows), len(rows)
            low, high = wilson(k, n)
            x = j + (i - (len(available) - 1) / 2) * width
            ax.bar(x, k / n, width=width * 0.92, color=color, edgecolor=SURFACE, linewidth=1.0, zorder=3,
                   label=name if j == 0 else None)
            ax.errorbar(x, k / n, yerr=[[k / n - low], [high - k / n]], fmt="none", ecolor=INK_2,
                        elinewidth=1.0, capsize=2, zorder=4)
            ax.text(x, high, f"{100 * k / n:.0f}", ha="center", va="bottom", fontsize=6, color=INK_2)
            points.append({"ha": name, "record_set": record_set, "n": n, "accuracy": k / n,
                           "ci_low": low, "ci_high": high})
    ax.set_xticks(range(len(HA_RECORD_SETS)), [f"{label}\n({s})" for s, label in HA_RECORD_SETS],
                  fontsize=8, color=INK_2)
    y_format(ax, "pct", 1)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=7, labelcolor=INK_2)
    save(fig, out_dir / "h4_ha_test", note="Error bars: 95% Wilson intervals. Same patients and records "
         "for every HA within a group. Primary H4a comparison: A_da2, CLM (h2) vs new HA.")
    write_points(out_dir / "h4_ha_test_points.csv", points)


# ------------------------------------------------------------ stage 4 validation curves

def plot_validation(out_dir: Path, chosen: dict[str, int]) -> None:
    from matplotlib.ticker import MultipleLocator

    runs = {name: load_validation(Path(path), 100) for name, path in TRAIN_RUNS if Path(path).is_dir()}
    runs = {name: pts for name, pts in runs.items() if pts}
    if not runs:
        print("  skipped h4_validation: no validation summaries")
        return
    colors = {name: SERIES[4 + i] for i, (name, _) in enumerate(TRAIN_RUNS)}  # same hues as the test figures
    max_step = max(p["step"] for pts in runs.values() for p in pts)
    fig, axes = new_figure(3, 2, 4.4, 2.9)
    for ax, (key, title, kind) in zip(axes.flat, VALIDATION_PANELS):
        base_style(ax, title)
        ax.set_xlim(-max_step * 0.02, max_step * 1.12)
        ax.xaxis.set_major_locator(MultipleLocator(100 if max_step <= 1000 else 200))
        if max_step > PREREGISTERED_STEPS:
            ax.axvline(PREREGISTERED_STEPS, color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
            ax.text(PREREGISTERED_STEPS, 1.0, "pre-registered end", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7, color=MUTED)
        top = 0.0
        for name, pts in runs.items():
            series = [(p["step"], p[key]) for p in pts if p.get(key) is not None]
            if not series:
                continue
            xs, ys = zip(*series)
            ax.plot(xs, ys, color=colors[name], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=name, clip_on=False, zorder=3)
            if name in chosen and chosen[name] in xs:
                y = ys[xs.index(chosen[name])]
                ax.scatter([chosen[name]], [y], s=150, facecolors="none", edgecolors=colors[name],
                           linewidths=1.8, zorder=4)
            top = max(top, max(ys))
        y_format(ax, kind, 1 if kind == "pct" else top)
    axes.flat[0].legend(loc="lower right", frameon=False, fontsize=7, labelcolor=INK_2)
    chosen_text = ", ".join(f"{n.split(' (')[0].lower()} step {s}" for n, s in chosen.items())
    save(fig, out_dir / "h4_validation",
         title="H4 validation: new DA trained with the CLM HA (240 cases x 4 sampled rollouts, every 100 steps)",
         note=f"Rings: each arm's step chosen on validation ({chosen_text}). Llama stays frozen in both arms; "
              "only the heads differ.")
    write_points(out_dir / "h4_validation_points.csv",
                 [{"run": name, "step": p["step"], **{k: p.get(k) for k, _, _ in VALIDATION_PANELS}}
                  for name, pts in runs.items() for p in pts])


# ------------------------------------------------------------ H4b and H4c test runs

def with_final_hypothesis(rows: list[dict]) -> list[dict]:
    """The same rows, scored by the HA's last hypothesis instead of the DA's diagnosis."""
    out = []
    for r in rows:
        guess = (r.get("final_hypothesis") or "").lower()
        out.append({**r, "diagnosis": guess if guess in DISEASES else "", "correct": guess == r["label"]})
    return out


def plot_test(out_dir: Path, resamples: int, runs_spec: list[tuple[str, str]]) -> None:
    runs = [(name, read_rows(Path(path) / "test_episodes.csv")) for name, path in runs_spec
            if (Path(path) / "test_episodes.csv").is_file()]
    missing = [f"{name} ({path})" for name, path in runs_spec if not (Path(path) / "test_episodes.csv").is_file()]
    if missing:
        print("  missing test runs:\n    " + "\n    ".join(missing))
    if not runs:
        return
    names = [name for name, _ in runs]
    palette = {name: SERIES[i] for i, (name, _) in enumerate(runs_spec)}
    colors = [palette[n] for n in names]
    ticks = [short(n) for n in names]
    da = {name: test_metrics(rows, resamples) for name, rows in runs}
    ha = {name: test_metrics(with_final_hypothesis(rows), resamples) for name, rows in runs}

    panels = [
        (da, "accuracy", "Accuracy: the DA's diagnosis", "pct"),
        (ha, "accuracy", "Accuracy: the HA's last hypothesis", "pct"),
        (da, "tests", "Average tests per case", "number"),
        (da, "cost", "Average test cost per case (USD)", "money"),
    ]
    fig, axes = new_figure(2, 2, 7.0, 3.8)
    for ax, (source, key, title, kind) in zip(axes.flat, panels):
        category_axis(ax, title, ticks)
        ax.tick_params(axis="x", labelsize=6.5)
        draw_ci_bars(ax, ticks, [source[n][key] for n in names], colors, kind)
    save(fig, out_dir / "h4_test_overview", title=f"Test set ({len(runs[0][1])} cases, 1 rollout each)",
         note="Error bars: 95% intervals (Wilson for accuracy; case bootstrap otherwise). "
              "Paired differences are in docs/HYPOTHESIS_H4.md.")
    write_points(out_dir / "h4_test_points.csv",
                 [{"run": n, "score": score, "metric": k, "value": v[0], "ci_low": v[1], "ci_high": v[2]}
                  for score, source in (("diagnosis", da), ("final_hypothesis", ha))
                  for n in names for k, v in source[n].items() if isinstance(v, tuple)])

    rates = {name: {} for name in names}
    for name, rows in runs:
        for r in rows:
            for test in {t.lower() for t in r["tests"]}:
                rates[name][test] = rates[name].get(test, 0) + 1 / len(rows)
    tests = sorted({t for r in rates.values() for t, v in r.items() if v >= 0.01},
                   key=lambda t: -max(rates[n].get(t, 0) for n in names))
    fig, axes = new_figure(1, 1, 12, 4.0)
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
    ax.legend(loc="upper right", frameon=False, fontsize=7, labelcolor=INK_2)
    save(fig, out_dir / "h4_test_rates", note="Tests ordered by fewer than 1% of patients in every run are left out.")
    write_points(out_dir / "h4_test_rates_points.csv",
                 [{"run": n, "test": t, "share": round(rates[n].get(t, 0), 4)} for n in names for t in tests])

    columns = 3
    fig, axes = new_figure((len(runs) + columns - 1) // columns, columns, 3.4, 3.6)
    for ax in axes.flat[len(runs):]:
        ax.set_visible(False)
    for ax, (name, rows) in zip(axes.flat, runs):
        matrix = []
        for true in DISEASES:
            group = [r for r in rows if r["label"] == true]
            matrix.append([sum((r["diagnosis"] or "undiagnosed") == pred for r in group) / len(group)
                           if group else 0 for pred in PREDICTED])
        draw_matrix(ax, matrix, name.replace(", ", ",\n"))
    for row in axes:
        row[0].set_ylabel("True disease", fontsize=8, color=INK_2)
    save(fig, out_dir / "h4_test_confusion", title="Test-set confusion matrices (the DA's diagnosis)",
         note=MATRIX_NOTE)

    from matplotlib.ticker import FuncFormatter

    fig, axes = new_figure(1, 1, 7, 4.6)
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
    ax.legend(loc="lower right", frameon=False, fontsize=7, labelcolor=INK_2)
    save(fig, out_dir / "h4_cost_vs_accuracy", note="Bars: 95% intervals. Up and to the left is better. "
         "The y-axis starts at 60%.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("figures/h4"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--joint-step", type=int, default=600, help="joint arm's validation-chosen step")
    parser.add_argument("--frozen-step", type=int, default=400, help="frozen arm's validation-chosen step")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_heads(args.out_dir)
    plot_ha_test(args.out_dir)
    plot_validation(args.out_dir, {TRAIN_RUNS[0][0]: args.joint_step, TRAIN_RUNS[1][0]: args.frozen_step})
    plot_test(args.out_dir, args.bootstrap, test_runs(args.joint_step, args.frozen_step))
    print(f"Wrote to {args.out_dir}/: h4_heads, h4_ha_test, h4_validation, h4_test_overview, h4_test_rates, "
          "h4_test_confusion, h4_cost_vs_accuracy")


if __name__ == "__main__":
    main()
