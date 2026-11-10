"""Plot what happened in the training rollouts, step by step.

Reads a run's ``train_episodes.csv`` (4 rollouts of one patient per step) and
plots per-step averages over windows of ``--window`` steps. Validation is only
measured every 100 steps; this shows the shape of learning in between, and the
GRPO learning signal that drives it.

    python scripts/plot_training_curve.py outputs/cost_sweep/cost_1/train \
        --out figures/training

Writes ``<out>.{pdf,png}`` and ``<out>_points.csv``. Decision Agent phases (the
only phases that train test ordering and diagnosis) are shaded. Rollouts are
sampled at the training temperature from the current model in every phase, so
the curves are continuous across phases; HA phases just do not change the DA.

Windows of 25 steps hold 100 rollouts of 25 different patients, so expect a
few points of noise between neighbouring windows.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_validation_curve import AXIS, CYCLE, GRID, INK, INK_2, MUTED, SERIES, SURFACE
from src.utils.episode_log import group_signal_flags, read_rows

DA_STEPS = 100  # first phase of every 300-step cycle

PANELS = [
    ("correct", "Rollouts diagnosed correctly", "pct"),
    ("undiagnosed", "Rollouts undiagnosed", "pct"),
    ("tests", "Tests that returned a result, per rollout", "number"),
    ("max_length", "Rollouts that hit the token limit", "pct_auto"),
    ("signal", "Groups with a learning signal", "pct"),
    ("cost_only", "Groups whose only signal was test cost", "pct_auto"),
]


def step_metrics(rows: list[dict]) -> dict[int, dict]:
    """One dict of metrics per training step (one GRPO group)."""
    by_step = defaultdict(list)
    for row in rows:
        by_step[row["step"]].append(row)
    out = {}
    for step, group in by_step.items():
        flags = group_signal_flags(group)
        n = len(group)
        out[step] = {
            "correct": sum(r["correct"] for r in group) / n,
            "undiagnosed": sum(not r["diagnosis"] for r in group) / n,
            "tests": sum(r["num_tests"] for r in group) / n,
            "max_length": sum(r["end_reason"] == "max_length" for r in group) / n,
            "signal": 1.0 - flags["no_signal"] if flags else None,
            "cost_only": flags.get("cost_only_signal") if flags else None,
        }
    return out


def windows(per_step: dict[int, dict], window: int) -> list[dict]:
    """Average step metrics over consecutive windows; x is the window's last step."""
    binned = defaultdict(list)
    for step, metrics in per_step.items():
        binned[(step - 1) // window].append(metrics)
    points = []
    for index in sorted(binned):
        group = binned[index]
        point = {"step": min((index + 1) * window, max(per_step)), "steps_in_window": len(group)}
        for key, _, _ in PANELS:
            values = [m[key] for m in group if m[key] is not None]
            point[key] = sum(values) / len(values) if values else None
        points.append(point)
    return points


def style(ax, title: str, kind: str, max_step: int, top: float) -> None:
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1 if kind == "pct" else (top * 1.15 if top > 0 else 1))
    if kind in ("pct", "pct_auto"):
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlim(0, max_step)
    ax.xaxis.set_major_locator(MultipleLocator(CYCLE))
    for start in range(0, max_step, CYCLE):
        ax.axvspan(start, min(start + DA_STEPS, max_step), color=GRID, alpha=0.45, linewidth=0, zorder=0)
    ax.set_xlabel("Patients trained on (steps)", fontsize=8, color=INK_2)


def plot(points: list[dict], path_stem: Path, window: int, title: str) -> None:
    import matplotlib.pyplot as plt

    max_step = max(p["step"] for p in points)
    columns = 3
    rows = (len(PANELS) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.8 * columns, 2.9 * rows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, panel_title, kind) in zip(axes.flat, PANELS):
        series = [(p["step"], p[key]) for p in points if p[key] is not None]
        top = max((y for _, y in series), default=0)
        style(ax, panel_title, kind, max_step, top)
        if not series:
            ax.text(0.5, 0.5, "not recorded", transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color=MUTED)
            continue
        xs, ys = zip(*series)
        ax.plot(xs, ys, color=SERIES[0], linewidth=1.5, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, f"Each point averages {window} steps ({4 * window} rollouts). "
             "Shaded: Decision Agent phases. A group has a learning signal when its 4 rollouts' rewards "
             "are not all equal.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    for suffix in (".pdf", ".png"):
        fig.savefig(path_stem.with_name(path_stem.name + suffix), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train_dir", type=Path, help="Run's train/ directory, its episodes/ directory, or the CSV.")
    parser.add_argument("--out", type=Path, default=Path("figures/training"))
    parser.add_argument("--window", type=int, default=25, help="Steps per plotted point (default 25).")
    parser.add_argument("--title", default="Training rollouts over time")
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        sys.exit("matplotlib is required: pip install --no-cache-dir matplotlib")

    path = args.train_dir
    if path.is_dir():
        path = path / "episodes" / "train_episodes.csv" if (path / "episodes").is_dir() else path / "train_episodes.csv"
    if not path.is_file():
        sys.exit(f"No training episodes CSV at {path}")
    rows = read_rows(path)
    if not rows:
        sys.exit(f"{path} has no rows")

    points = windows(step_metrics(rows), args.window)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(points, args.out, args.window, args.title)
    keys = ["step", "steps_in_window"] + [k for k, _, _ in PANELS]
    with args.out.with_name(args.out.name + "_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(points)
    print(f"{len(rows)} rollouts over {len({r['step'] for r in rows})} steps")
    print(f"Wrote {args.out}.pdf/.png and {args.out}_points.csv")


if __name__ == "__main__":
    main()
