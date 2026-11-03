"""Verify and compare TRACE-CDM evaluation arms.

Each ``--run`` points at an evaluation output directory containing the fork's
``metrics.json`` and, for TRACE arms, ``trace_episodes.jsonl`` and
``trace_summary.json``.

Verification (fails with exit code 1 unless --no-strict):
  1. every arm's exported episodes reproduce its own metrics.json;
  2. the shadow arm's metrics.json matches the native baseline's;
  3. all shadow arms (shadow, shadow_adaptive) take identical actions.

Then every arm is compared with the reference arm (default: shadow, the
instrumented reproduction of the baseline) on the same patients.

Example:
    python scripts/trace_compare.py \\
        --run baseline=outputs/trace_cdm/baseline/eval_123 \\
        --run shadow=outputs/trace_cdm/shadow/eval_124 \\
        --run deterministic=outputs/trace_cdm/deterministic/eval_125 \\
        --output outputs/trace_cdm/comparison
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.trace_cdm.metrics import (  # noqa: E402
    arm_metrics,
    load_episodes,
    paired_comparison,
    verify_against_metrics,
    verify_identical,
)

SHADOW_ARMS = ("shadow", "shadow_adaptive")
# Native metrics that must be identical between the baseline and the shadow arm.
NATIVE_KEYS = ("eval_all/accuracy", "eval_none_fraction", "eval_avg_num_tests", "eval_avg_diagnostic_cost",
               "eval_all/macro_f1", "eval_hypothesis_accuracy", "eval_ece")


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run expects NAME=DIR")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="NAME=EVAL_OUTPUT_DIR")
    parser.add_argument("--reference", default="shadow", help="arm the others are compared against")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-strict", action="store_true", help="report verification failures without failing")
    args = parser.parse_args()

    runs = dict(args.run)
    problems: dict[str, list[str]] = {}
    episodes, native, summaries = {}, {}, {}
    for name, run_dir in runs.items():
        metrics_file = run_dir / "metrics.json"
        if not metrics_file.exists():
            problems.setdefault(name, []).append(f"missing {metrics_file}")
            continue
        native[name] = json.loads(metrics_file.read_text())
        if (run_dir / "trace_episodes.jsonl").exists():
            episodes[name] = load_episodes(run_dir / "trace_episodes.jsonl")
            mismatch = verify_against_metrics(episodes[name], metrics_file)
            if mismatch:
                problems.setdefault(name, []).extend(f"episodes vs metrics.json: {m}" for m in mismatch)
        if (run_dir / "trace_summary.json").exists():
            summaries[name] = json.loads((run_dir / "trace_summary.json").read_text())

    if "baseline" in native and "shadow" in native:
        for key in NATIVE_KEYS:
            a, b = native["baseline"].get(key), native["shadow"].get(key)
            if a != b:
                problems.setdefault("shadow", []).append(f"{key}: baseline={a} shadow={b}")
    shadow_present = [n for n in SHADOW_ARMS if n in episodes]
    for other in shadow_present[1:]:
        diffs = verify_identical(episodes[shadow_present[0]], episodes[other])
        if diffs:
            problems.setdefault(other, []).extend(f"vs {shadow_present[0]}: {d}" for d in diffs[:20])

    report = {"reference": args.reference, "verification_problems": problems, "arms": {}, "paired_vs_reference": {}}
    for name in runs:
        if name in episodes:
            report["arms"][name] = arm_metrics(episodes[name])
        elif name in native:
            report["arms"][name] = {k: native[name].get(k) for k in NATIVE_KEYS}
        if name in summaries:
            report["arms"].setdefault(name, {})["trace_summary"] = summaries[name]
    if args.reference in episodes:
        for name in episodes:
            if name != args.reference:
                report["paired_vs_reference"][name] = paired_comparison(
                    episodes[args.reference], episodes[name], args.bootstrap, args.seed)
    elif len(episodes) > 1:
        problems.setdefault(args.reference, []).append("reference arm has no exported episodes; no paired comparison")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2))
    (args.output / "comparison.md").write_text(render_markdown(report))
    print((args.output / "comparison.md").read_text())

    if problems:
        print("VERIFICATION FAILED:", file=sys.stderr)
        for name, items in problems.items():
            for item in items:
                print(f"  [{name}] {item}", file=sys.stderr)
        return 0 if args.no_strict else 1
    return 0


def _fmt(value, pct=False):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{100 * value:.2f}%" if pct else f"{value:.3f}"
    return str(value)


def render_markdown(report: dict) -> str:
    lines = ["# TRACE-CDM arm comparison", "",
             "| Arm | Cases | Accuracy | Undiagnosed | Mean tests | Mean cost (USD) | Critic calls / decision |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in report["arms"].items():
        if "episodes" in m:
            lines.append(f"| {name} | {m['episodes']} | {_fmt(m['accuracy'], True)} | {_fmt(m['undiagnosed_fraction'], True)} "
                         f"| {_fmt(m['mean_tests'])} | {_fmt(m['mean_cost_usd'])} | {_fmt(m['critic_call_rate'])} |")
        else:
            lines.append(f"| {name} (native) | - | {_fmt(m.get('eval_all/accuracy'), True)} | {_fmt(m.get('eval_none_fraction'), True)} "
                         f"| {_fmt(m.get('eval_avg_num_tests'))} | {_fmt(m.get('eval_avg_diagnostic_cost'))} | - |")
    if report["paired_vs_reference"]:
        lines += ["", f"## Paired differences vs `{report['reference']}` (arm minus reference, bootstrap 95% CI)", "",
                  "| Arm | Paired | Δ accuracy | Δ tests | Δ cost (USD) | ref✓ arm✗ | ref✗ arm✓ | McNemar p |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, p in report["paired_vs_reference"].items():
            if not p.get("paired_cases"):
                continue
            acc_lo, acc_hi = p["delta_accuracy_ci95"]
            lines.append(
                f"| {name} | {p['paired_cases']} | {100 * p['delta_accuracy']:+.2f} pp [{100 * acc_lo:+.2f}, {100 * acc_hi:+.2f}] "
                f"| {p['delta_tests']:+.2f} [{p['delta_tests_ci95'][0]:+.2f}, {p['delta_tests_ci95'][1]:+.2f}] "
                f"| {p['delta_cost_usd']:+.1f} [{p['delta_cost_usd_ci95'][0]:+.1f}, {p['delta_cost_usd_ci95'][1]:+.1f}] "
                f"| {p['reference_correct_arm_wrong']} | {p['reference_wrong_arm_correct']} | {p['mcnemar_exact_p']:.3g} |")
    problems = report["verification_problems"]
    lines += ["", "## Verification", ""]
    lines += [f"- [{n}] {i}" for n, items in problems.items() for i in items] or ["- all checks passed"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
