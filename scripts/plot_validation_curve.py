"""Plot validation results over training steps for one or more runs.

Each ``--run`` is ``LABEL=PATH``. PATH is either a run's ``train/`` directory
(reads the ``val_step_*_summary.json`` files written since November 2026) or a
Slurm training log from an older run (reads the validation dictionaries it
printed; those runs validated every ``--log-eval-steps`` steps and did not
record test cost).

    python scripts/plot_validation_curve.py \
        --run 1x=outputs/cost_sweep/cost_1/train \
        --run 0x=logs/lacdm_cost_train_48885778.out \
        --run 1.5x=logs/lacdm_cost_train_49166794.out \
        --out figures/validation

Writes ``<out>_overview.{pdf,png}`` (overall accuracy, undiagnosed share and
cost per case) and ``<out>_per_disease.{pdf,png}`` (accuracy for each disease),
plus ``<out>_points.csv`` with every plotted value. The untrained model is the
same for every arm, so a step-0 point measured in one run is shown for all
runs that lack one (disable with ``--no-shared-step0``).
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validation_curve import load_curve
from src.utils.episode_log import DISEASES

# Reference categorical palette (light mode), assigned to runs in the order given.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CYCLE, PHASES = 300, ("DA", "HA-SFT", "Calib.")


def points_from_log(path: Path, eval_steps: int) -> list[dict]:
    """Parse the validation dictionaries a training job printed, in order."""
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


def load_run(path: Path, eval_steps: int) -> list[dict]:
    if path.is_dir():
        return load_curve(path)
    return points_from_log(path, eval_steps)


def style_axis(ax, title: str, percent: bool, money: bool, max_step: int) -> None:
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=14)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    if percent:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    elif money:
        # The y-range is set after plotting, from the data.
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_xlim(-max_step * 0.02, max_step * 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(100 if max_step <= 900 else CYCLE))
    # Phase boundaries: faint every 100 steps, slightly stronger at cycle ends.
    for step in range(100, max_step + 1, 100):
        ax.axvline(step, color=GRID if step % CYCLE else AXIS, linewidth=0.8, zorder=0)
    if max_step <= 900:
        for start in range(0, max_step, 100):
            ax.text(start + 50, 1.0, PHASES[(start // 100) % 3], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7, color=MUTED)


def draw(ax, runs: dict, key: str, colors: dict, label_ends: bool) -> bool:
    drawn = False
    for name, points in runs.items():
        series = [(p["step"], p[key]) for p in points if p.get(key) is not None]
        if not series:
            continue
        xs, ys = zip(*series)
        ax.plot(xs, ys, color=colors[name], linewidth=1.5, solid_capstyle="round", solid_joinstyle="round",
                marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name,
                clip_on=False, zorder=3)
        # A lone point (e.g. only the shared step 0) is identified by the legend.
        if label_ends and len(series) >= 2:
            ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
        drawn = True
    return drawn


def figure(runs, colors, panels, path_stem: Path, columns: int, max_step: int) -> None:
    import matplotlib.pyplot as plt

    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 2.9 * rows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, title, percent, money) in zip(axes.flat, panels):
        style_axis(ax, title, percent, money, max_step)
        if not draw(ax, runs, key, colors, label_ends=len(runs) <= 4):
            ax.text(0.5, 0.5, "not recorded for these runs", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color=MUTED)
            ax.set_yticks([])
        elif money:
            top = max(p[key] for points in runs.values() for p in points if p.get(key) is not None)
            ax.set_ylim(0, top * 1.15)
        ax.set_xlabel("Patients trained on (steps)", fontsize=8, color=INK_2)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    if len(runs) >= 2:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=len(runs), frameon=False, fontsize=8,
                   labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if len(runs) >= 2 else 1))
    for suffix in (".pdf", ".png"):
        fig.savefig(path_stem.with_name(path_stem.name + suffix), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--out", type=Path, default=Path("figures/validation"))
    parser.add_argument("--log-eval-steps", type=int, default=100)
    parser.add_argument("--no-shared-step0", action="store_true")
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        sys.exit("matplotlib is required: pip install --no-cache-dir matplotlib")

    runs = {}
    for item in args.run:
        name, _, path = item.partition("=")
        if not path:
            sys.exit(f"--run must be LABEL=PATH, got {item!r}")
        points = load_run(Path(path), args.log_eval_steps)
        if not points:
            sys.exit(f"No validation results found in {path}")
        runs[name] = points
    if len(runs) > len(SERIES):
        sys.exit(f"At most {len(SERIES)} runs per chart")
    if len(runs) > 3:
        print("Note: more than three runs makes lines hard to tell apart; consider separate charts.")

    if not args.no_shared_step0:
        step0 = next((p for points in runs.values() for p in points if p["step"] == 0), None)
        if step0 is not None:
            for name, points in runs.items():
                if not any(p["step"] == 0 for p in points):
                    points.insert(0, step0)
                    print(f"{name}: using the shared step-0 (untrained) validation point")

    colors = {name: SERIES[i] for i, name in enumerate(runs)}
    max_step = max(p["step"] for points in runs.values() for p in points)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure(runs, colors, [
        ("accuracy", "Accuracy (undiagnosed counted wrong)", True, False),
        ("undiagnosed", "Undiagnosed share", True, False),
        ("cost_usd", "Test cost per case (USD)", False, True),
    ], args.out.with_name(args.out.name + "_overview"), columns=3, max_step=max_step)
    figure(runs, colors, [
        (f"accuracy_{d}", f"{d.capitalize()}: accuracy", True, False) for d in DISEASES
    ], args.out.with_name(args.out.name + "_per_disease"), columns=2, max_step=max_step)

    keys = ["step", "accuracy", "undiagnosed", "cost_usd"] + [f"accuracy_{d}" for d in DISEASES]
    with args.out.with_name(args.out.name + "_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run"] + keys)
        for name, points in runs.items():
            for point in points:
                writer.writerow([name] + [point.get(k) for k in keys])
    print(f"Wrote {args.out}_overview.pdf/.png, {args.out}_per_disease.pdf/.png, {args.out}_points.csv")


if __name__ == "__main__":
    main()
