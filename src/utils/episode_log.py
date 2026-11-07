"""Per-episode records, CSV persistence, and per-disease summaries.

One row describes one episode: a single rollout of the Decision Agent and
Hypothesis Agent on one patient. Training writes one row per GRPO rollout,
validation one row per validation rollout, and test evaluation one row per
test case. The module uses only the standard library so it can be tested
without the training stack.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

DISEASES = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")
UNDIAGNOSED = "undiagnosed"
END_REASONS = ("diagnosis", "invalid_format", "max_steps", "max_length")
LIST_SEPARATOR = "|"

# Column order of every episode CSV. List-valued fields are joined with "|".
EPISODE_FIELDS = (
    "stage",
    "step",
    "phase",
    "patient_id",
    "episode",
    "label",
    "diagnosis",
    "correct",
    "end_reason",
    "num_turns",
    "num_tests",
    "tests",
    "cost_usd",
    "actions",
    "num_unavailable_requests",
    "num_unknown_test_requests",
    "num_repeat_requests",
    "num_unknown_actions",
    "num_unknown_diagnoses",
    "hypotheses",
    "confidences",
    "final_hypothesis",
    "final_confidence",
    "reward_diagnosis",
    "reward_format",
    "reward_test_cost",
    "reward_total",
    "advantage",
    "transcript_tokens",
    "last_completion",
)

_LIST_FIELDS = {"tests", "actions", "hypotheses", "confidences"}
_INT_FIELDS = {
    "step",
    "episode",
    "num_turns",
    "num_tests",
    "num_unavailable_requests",
    "num_unknown_test_requests",
    "num_repeat_requests",
    "num_unknown_actions",
    "num_unknown_diagnoses",
    "transcript_tokens",
}
_FLOAT_FIELDS = {
    "cost_usd",
    "final_confidence",
    "reward_diagnosis",
    "reward_format",
    "reward_test_cost",
    "reward_total",
    "advantage",
}


def build_episode_rows(
    records: list[dict],
    *,
    stage: str,
    step: int,
    phase: str = "",
    reward_names: list[str] | None = None,
    rewards_per_func: list[list[float]] | None = None,
    advantages: list[float] | None = None,
    test_costs_usd: dict[str, float] | None = None,
) -> list[dict]:
    """Combine environment records with rewards into typed episode rows.

    Args:
        records: ``EnvironmentOutput.episode_records``, one per rollout.
        stage: ``"train"``, ``"val"``, or ``"test"``.
        step: Optimizer step the episode belongs to.
        phase: Active loss phase during training, empty otherwise.
        reward_names: Reward-function names, aligned with ``rewards_per_func`` columns.
        rewards_per_func: Per-rollout reward components (rows align with ``records``).
        advantages: Per-rollout GRPO advantages; only meaningful during training.
        test_costs_usd: Real-world USD price per test (lower-case names).
    """
    reward_names = reward_names or []
    test_costs_usd = {k.lower(): v for k, v in (test_costs_usd or {}).items()}
    episode_counter: Counter = Counter()
    rows = []
    for index, record in enumerate(records):
        patient_id = record["patient_id"]
        components = dict(zip(reward_names, rewards_per_func[index])) if rewards_per_func else {}
        diagnosis = record.get("diagnosis") or ""
        confidences = record.get("confidences") or []
        hypotheses = record.get("hypotheses") or []
        tests = list(record.get("provided_tests") or [])
        row = {
            "stage": stage,
            "step": int(step),
            "phase": phase,
            "patient_id": patient_id,
            "episode": episode_counter[patient_id],
            "label": str(record["label"]).lower(),
            "diagnosis": diagnosis.lower(),
            "correct": bool(diagnosis) and diagnosis.lower() == str(record["label"]).lower(),
            "end_reason": record.get("end_reason") or "",
            "num_turns": int(record.get("num_turns", 0)),
            "num_tests": len(tests),
            "tests": tests,
            "cost_usd": float(sum(test_costs_usd.get(t.lower(), 0.0) for t in tests)),
            "actions": list(record.get("actions") or []),
            "num_unavailable_requests": int(record.get("num_unavailable_requests", 0)),
            "num_unknown_test_requests": int(record.get("num_unknown_test_requests", 0)),
            "num_repeat_requests": int(record.get("num_repeat_requests", 0)),
            "num_unknown_actions": int(record.get("num_unknown_actions", 0)),
            "num_unknown_diagnoses": int(record.get("num_unknown_diagnoses", 0)),
            "hypotheses": list(hypotheses),
            "confidences": confidences,
            "final_hypothesis": hypotheses[-1] if hypotheses else "",
            "final_confidence": next((c for c in reversed(confidences) if c is not None), None),
            "reward_diagnosis": components.get("DiagnosisReward"),
            "reward_format": components.get("FormatReward"),
            "reward_test_cost": components.get("TestCostReward"),
            "reward_total": sum(components.values()) if components else None,
            "advantage": advantages[index] if advantages is not None else None,
            "transcript_tokens": int(record.get("transcript_tokens", 0)),
            "last_completion": record.get("last_completion") or "",
        }
        episode_counter[patient_id] += 1
        rows.append(row)
    return rows


def _encode(field: str, value) -> str:
    if value is None:
        return ""
    if field in _LIST_FIELDS:
        return LIST_SEPARATOR.join("" if v is None else _format_number(v) for v in value)
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return _format_number(value)
    return str(value)


def _format_number(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _decode(field: str, value: str):
    if field in _LIST_FIELDS:
        if value == "":
            return []
        items = value.split(LIST_SEPARATOR)
        if field == "confidences":
            return [float(v) if v != "" else None for v in items]
        return items
    if field == "correct":
        return value == "1"
    if value == "":
        return None if field in _INT_FIELDS | _FLOAT_FIELDS else ""
    if field in _INT_FIELDS:
        return int(value)
    if field in _FLOAT_FIELDS:
        return float(value)
    return value


def append_rows(path: Path, rows: Iterable[dict]) -> None:
    """Append rows to a CSV, writing the header when the file is new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: _encode(field, row.get(field)) for field in EPISODE_FIELDS})


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    """Write rows to a new CSV, replacing any existing file."""
    path = Path(path)
    if path.exists():
        path.unlink()
    append_rows(path, rows)


def read_rows(path: Path) -> list[dict]:
    """Read an episode CSV back into typed rows."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [
            {field: _decode(field, row.get(field, "")) for field in EPISODE_FIELDS}
            for row in csv.DictReader(handle)
        ]


def truncate_after_step(path: Path, max_step: int) -> int:
    """Drop rows whose step exceeds ``max_step``; return the number removed.

    A resumed job replays every step after its checkpoint, so rows written
    after that checkpoint would otherwise appear twice.
    """
    path = Path(path)
    if not path.exists():
        return 0
    rows = read_rows(path)
    kept = [row for row in rows if row["step"] is not None and row["step"] <= max_step]
    if len(kept) != len(rows):
        write_rows(path, kept)
    return len(rows) - len(kept)


def write_transcripts(path: Path, rows: list[dict], records: list[dict]) -> None:
    """Write full episode transcripts as JSON lines aligned with ``rows``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row, record in zip(rows, records):
            handle.write(
                json.dumps(
                    {
                        "patient_id": row["patient_id"],
                        "episode": row["episode"],
                        "label": row["label"],
                        "diagnosis": row["diagnosis"],
                        "end_reason": row["end_reason"],
                        "transcript": record.get("transcript", ""),
                    }
                )
                + "\n"
            )


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def expected_calibration_error(rows: list[dict]) -> float | None:
    """ECE of Hypothesis Agent confidence over every HA call.

    Confidences are integers 0-10, so each value is its own bin, matching the
    trainer's 0.1-wide bins. Calls with an unknown hypothesis or missing
    confidence are excluded, as in ``calculate_classification_metrics``.
    """
    bins: dict[float, list[bool]] = defaultdict(list)
    for row in rows:
        for hypothesis, confidence in zip(row["hypotheses"], row["confidences"]):
            if confidence is None or hypothesis in ("", "unknown") or not 0 <= confidence <= 10:
                continue
            bins[float(confidence)].append(hypothesis == row["label"])
    total = sum(len(outcomes) for outcomes in bins.values())
    if total == 0:
        return None
    return sum(
        len(outcomes) / total * abs(confidence / 10 - sum(outcomes) / len(outcomes))
        for confidence, outcomes in bins.items()
    )


def _group_summary(rows: list[dict], tests: list[str]) -> dict:
    n = len(rows)
    diagnosed = [r for r in rows if r["diagnosis"]]
    hypotheses = [h for r in rows for h in r["hypotheses"]]
    hypothesis_correct = sum(
        1 for r in rows for h in r["hypotheses"] if h == r["label"]
    )
    end_reasons = Counter(r["end_reason"] for r in rows)
    test_counts = Counter(t.lower() for r in rows for t in set(r["tests"]))
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n if n else None,
        "accuracy_diagnosed": (
            sum(r["correct"] for r in diagnosed) / len(diagnosed) if diagnosed else None
        ),
        "undiagnosed_fraction": (n - len(diagnosed)) / n if n else None,
        "avg_num_tests": _mean([r["num_tests"] for r in rows]),
        "avg_cost_usd": _mean([r["cost_usd"] for r in rows]),
        "avg_turns": _mean([r["num_turns"] for r in rows]),
        "hypothesis_accuracy": hypothesis_correct / len(hypotheses) if hypotheses else None,
        "ece": expected_calibration_error(rows),
        "final_hypothesis_accuracy": (
            sum(1 for r in rows if r["final_hypothesis"] == r["label"]) / n if n else None
        ),
        "end_reasons": {reason: end_reasons.get(reason, 0) / n if n else 0.0 for reason in END_REASONS},
        "test_rates": {test: test_counts.get(test, 0) / n if n else 0.0 for test in tests},
    }


def summarize(rows: list[dict], diseases: Iterable[str] = DISEASES, tests: Iterable[str] = ()) -> dict:
    """Summarize episodes overall, per true disease, and as a confusion matrix.

    Per-disease ``accuracy`` is recall for that disease with undiagnosed
    episodes counted wrong. ``precision`` and ``f1`` use all predictions of the
    disease. ``test_rates`` is the fraction of episodes in which each test
    returned a result.
    """
    diseases = [d.lower() for d in diseases]
    tests = [t.lower() for t in tests] or sorted({t.lower() for r in rows for t in r["tests"]})
    summary = {"overall": _group_summary(rows, tests), "per_disease": {}}
    confusion = {d: {p: 0 for p in diseases + [UNDIAGNOSED]} for d in diseases}
    for row in rows:
        if row["label"] in confusion:
            prediction = row["diagnosis"] if row["diagnosis"] in diseases else UNDIAGNOSED
            confusion[row["label"]][prediction] += 1
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    for disease in diseases:
        group = _group_summary(by_label.get(disease, []), tests)
        predicted = sum(confusion[label][disease] for label in diseases)
        true_positive = confusion[disease][disease]
        recall = group["accuracy"]
        precision = true_positive / predicted if predicted else None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0 if group["n"] else None
        group.update({"precision": precision, "f1": f1, "num_predicted": predicted})
        summary["per_disease"][disease] = group
    summary["confusion"] = confusion
    f1_values = [summary["per_disease"][d]["f1"] for d in diseases if summary["per_disease"][d]["f1"] is not None]
    summary["overall"]["macro_f1"] = sum(f1_values) / len(f1_values) if f1_values else None
    return summary


def flatten_summary(summary: dict, prefix: str) -> dict[str, float]:
    """Flatten a summary into numeric metric keys for Trainer logs."""
    flat: dict[str, float] = {}

    def add(key: str, value) -> None:
        if isinstance(value, bool):
            value = float(value)
        if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
            flat[key] = float(value)

    for key, value in summary["overall"].items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                add(f"{prefix}{key}/{sub_key}", sub_value)
        else:
            add(f"{prefix}episodes/{key}", value)
    for disease, group in summary["per_disease"].items():
        for key, value in group.items():
            if key == "test_rates":
                continue  # Per-disease test rates stay in the JSON summary.
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    add(f"{prefix}disease/{disease}/{key}/{sub_key}", sub_value)
            else:
                add(f"{prefix}disease/{disease}/{key}", value)
    for label, predictions in summary["confusion"].items():
        for prediction, count in predictions.items():
            add(f"{prefix}confusion/{label}/{prediction}", count)
    return flat


def group_signal_flags(rows: list[dict]) -> dict[str, float]:
    """Describe the GRPO learning signal for one training group.

    ``cost_only_signal`` is 1 when every rollout received the same diagnosis
    and format reward, so all advantage variation came from test cost.
    ``no_signal`` is 1 when all rollouts received the same total reward.
    """
    if not rows or rows[0]["reward_total"] is None:
        return {}
    outcome = {(r["reward_diagnosis"], r["reward_format"]) for r in rows}
    totals = [r["reward_total"] for r in rows]
    no_signal = max(totals) - min(totals) < 1e-9
    return {
        "correct_fraction": sum(r["correct"] for r in rows) / len(rows),
        "invalid_fraction": sum(not r["diagnosis"] for r in rows) / len(rows),
        "cost_only_signal": float(len(outcome) == 1 and not no_signal),
        "no_signal": float(no_signal),
    }
