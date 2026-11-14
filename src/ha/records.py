"""Turn four-way HA scores into episode rows (read by scripts/test_stats.py) and summaries.

Standard library only.
"""

from __future__ import annotations

import math

from src.ha.prompts import DISEASES


def scored_row(source: dict, probs: list[float]) -> dict:
    """Episode row for one patient: the source arm's tests, the HA's top disease as the answer.

    ``source`` is a row of the arm's ``test_episodes.csv`` (it fixes the tests and their cost).
    The confidence is the top probability on the agents' 0-10 scale, as test_stats expects.
    """
    best = max(range(len(DISEASES)), key=lambda i: probs[i])
    prediction, label = DISEASES[best], str(source["label"]).lower()
    confidence = round(10 * probs[best])
    tests = list(source["tests"])
    return {
        "stage": "test",
        "step": 0,
        "phase": "",
        "patient_id": source["patient_id"],
        "episode": 0,
        "label": label,
        "diagnosis": prediction,
        "correct": prediction == label,
        "end_reason": "diagnosis",
        "num_turns": 1,
        "num_tests": len(tests),
        "tests": tests,
        "cost_usd": source["cost_usd"],
        "actions": [f"test:{t}" for t in tests] + [f"diagnosis:{prediction}"],
        "hypotheses": [prediction],
        "confidences": [confidence],
        "final_hypothesis": prediction,
        "final_confidence": confidence,
        "last_completion": f"Hypothesis: {prediction}",
    }


def _auroc(scores: list[float], positives: list[bool]) -> float | None:
    pos = [s for s, p in zip(scores, positives) if p]
    neg = [s for s, p in zip(scores, positives) if not p]
    if not pos or not neg:
        return None
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def probability_summary(labels: list[str], probs: list[list[float]], bins: int = 10) -> dict:
    """Metrics that need the full distribution: exact AUROCs, log loss, Brier, binned ECE."""
    n = len(labels)
    truth = [DISEASES.index(l) for l in labels]
    top = [max(range(len(DISEASES)), key=lambda i: p[i]) for p in probs]
    correct = [t == k for t, k in zip(top, truth)]
    confidence = [p[k] for p, k in zip(probs, top)]
    per_class = {d: _auroc([p[i] for p in probs], [t == i for t in truth]) for i, d in enumerate(DISEASES)}
    valid = [a for a in per_class.values() if a is not None]
    ece = 0.0
    for b in range(bins):
        members = [i for i, c in enumerate(confidence) if (b / bins < c <= (b + 1) / bins) or (b == 0 and c == 0)]
        if members:
            ece += len(members) / n * abs(sum(confidence[i] for i in members) / len(members)
                                          - sum(correct[i] for i in members) / len(members))
    return {
        "n": n,
        "accuracy": sum(correct) / n,
        "auroc_ovr_exact": sum(valid) / len(valid) if valid else None,
        "auroc_per_disease": per_class,
        "auroc_correct_exact": _auroc(confidence, correct),
        "log_loss": -sum(math.log(max(p[t], 1e-12)) for p, t in zip(probs, truth)) / n,
        "brier": sum(sum((p[i] - (i == t)) ** 2 for i in range(len(DISEASES))) for p, t in zip(probs, truth)) / n,
        "ece_10_bins": ece,
        "mean_confidence": sum(confidence) / n,
        "recall": {d: (sum(c for c, t in zip(correct, truth) if t == i) / max(1, sum(t == i for t in truth)))
                   for i, d in enumerate(DISEASES)},
    }
