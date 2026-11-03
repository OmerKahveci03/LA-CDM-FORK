"""Patient-paired comparison of TRACE arms from exported episodes."""

import json
import math
import random
from pathlib import Path

CLASSES = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")


def load_episodes(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def arm_metrics(episodes: list[dict], classes=CLASSES) -> dict:
    n = len(episodes)
    if n == 0:
        return {"episodes": 0}
    correct = [e["diagnosis"] == e["ground_truth"] for e in episodes]
    out = {
        "episodes": n,
        "accuracy": sum(correct) / n,
        "undiagnosed_fraction": sum(e["diagnosis"] is None for e in episodes) / n,
        "invalid_fraction": sum(bool(e["invalid"]) for e in episodes) / n,
        "mean_tests": sum(e["num_tests"] for e in episodes) / n,
        "mean_cost_usd": sum(e["cost_usd"] for e in episodes) / n,
        "decision_points": sum(len(e["steps"]) for e in episodes),
        "critic_calls": sum(e.get("critic_calls", 0) for e in episodes),
    }
    out["critic_calls_per_episode"] = out["critic_calls"] / n
    out["critic_call_rate"] = out["critic_calls"] / out["decision_points"] if out["decision_points"] else 0.0
    diagnosed = [c for c, e in zip(correct, episodes) if e["diagnosis"] is not None]
    out["accuracy_among_diagnosed"] = sum(diagnosed) / len(diagnosed) if diagnosed else None
    for c in classes:
        cases = [ok for ok, e in zip(correct, episodes) if e["ground_truth"] == c]
        out[f"recall_{c}"] = sum(cases) / len(cases) if cases else None
    return out


def _pair(reference: list[dict], other: list[dict]) -> list[tuple[dict, dict]]:
    ref = {e["case_id"]: e for e in reference}
    oth = {e["case_id"]: e for e in other}
    return [(ref[c], oth[c]) for c in sorted(ref.keys() & oth.keys())]


def _bootstrap_ci(diffs: list[float], n_boot: int, rng: random.Random) -> tuple[float, float]:
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[min(n_boot - 1, int(0.975 * n_boot))]


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_comparison(reference: list[dict], other: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Arm minus reference on the cases both arms evaluated, with bootstrap 95% CIs."""
    pairs = _pair(reference, other)
    if not pairs:
        return {"paired_cases": 0}
    rng = random.Random(seed)
    acc = [float(o["diagnosis"] == o["ground_truth"]) - float(r["diagnosis"] == r["ground_truth"]) for r, o in pairs]
    cost = [o["cost_usd"] - r["cost_usd"] for r, o in pairs]
    tests = [o["num_tests"] - r["num_tests"] for r, o in pairs]
    b = sum(1 for d in acc if d < 0)  # reference right, arm wrong
    c = sum(1 for d in acc if d > 0)  # reference wrong, arm right
    result = {"paired_cases": len(pairs), "reference_correct_arm_wrong": b, "reference_wrong_arm_correct": c,
              "mcnemar_exact_p": mcnemar_exact_p(b, c)}
    for name, diffs in (("accuracy", acc), ("cost_usd", cost), ("tests", tests)):
        lo, hi = _bootstrap_ci(diffs, n_boot, rng)
        result[f"delta_{name}"] = sum(diffs) / len(diffs)
        result[f"delta_{name}_ci95"] = [lo, hi]
    result["unpaired_cases"] = len({e["case_id"] for e in reference} ^ {e["case_id"] for e in other})
    return result


def action_sequence(episode: dict) -> list[tuple]:
    return [(s["action"], s["action_input"], s["outcome"]) for s in episode["steps"]]


def verify_identical(reference: list[dict], other: list[dict]) -> list[str]:
    """Differences in case set, diagnoses or action sequences (empty list = identical)."""
    problems = []
    ref_ids = [e["case_id"] for e in reference]
    oth_ids = [e["case_id"] for e in other]
    if sorted(ref_ids) != sorted(oth_ids):
        problems.append(f"case sets differ: {len(set(ref_ids) ^ set(oth_ids))} cases not shared")
    for r, o in _pair(reference, other):
        if r["diagnosis"] != o["diagnosis"]:
            problems.append(f"{r['case_id']}: diagnosis {r['diagnosis']!r} vs {o['diagnosis']!r}")
        elif action_sequence(r) != action_sequence(o):
            problems.append(f"{r['case_id']}: action sequences differ")
    return problems


def verify_against_metrics(episodes: list[dict], metrics_path, tol: float = 1e-9) -> list[str]:
    """Check exported episodes reproduce the fork's own metrics.json for the same run."""
    metrics = json.loads(Path(metrics_path).read_text())
    mine = arm_metrics(episodes)
    checks = {
        "eval_all/accuracy": mine["accuracy"],
        "eval_none_fraction": mine["undiagnosed_fraction"],
        "eval_avg_num_tests": mine["mean_tests"],
        "eval_avg_diagnostic_cost": mine["mean_cost_usd"],
    }
    problems = []
    for key, value in checks.items():
        if key not in metrics:
            problems.append(f"{key} missing from {metrics_path}")
        elif abs(metrics[key] - value) > tol:
            problems.append(f"{key}: metrics.json={metrics[key]} episodes={value}")
    return problems
