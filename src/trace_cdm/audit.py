"""Deterministic, read-only evidence audit at each decision point.

Checks are restricted to what MIMIC-CDM actually exposes: the history summary,
the results of tests the agent has requested so far, and the Hypothesis Agent's
hypothesis/confidence. The dataset has no per-result timestamps, so temporal
leakage is prevented structurally (evidence is built only from requested tests)
rather than by comparing times.
"""

import re

from src.trace_cdm.schemas import (
    NOT_AVAILABLE,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    AuditResult,
    EvidenceItem,
    Flag,
    StepState,
)

# Hedging phrases that make an imaging or exam finding equivocal.
AMBIGUITY_TERMS = (
    "cannot be excluded",
    "can not be excluded",
    "cannot be ruled out",
    "not excluded",
    "equivocal",
    "indeterminate",
    "questionable",
    "possibly",
    "possible",
    "suspicious for",
    "may represent",
    "limited evaluation",
    "limited study",
    "not well visualized",
    "not visualized",
    "nondiagnostic",
    "non-diagnostic",
    "suboptimal",
)

# Placeholder values that carry no measurement ("___" is MIMIC de-identification).
INVALID_LAB_VALUES = {"", "nan", "none", "null", "___"}

ACTION_ERROR_OUTCOMES = {"unknown_action", "unknown_test", "unknown_diagnosis"}

_LAB_PAIR = re.compile(r"([^,:\n]+):\s*([^,\n]*)")


def source_map(lab_tests, imaging_tests, other_tests) -> dict[str, str]:
    """Map lower-cased test names to an evidence source category."""
    sources = {t.lower(): "lab" for t in lab_tests}
    sources.update({t.lower(): "imaging" for t in imaging_tests})
    sources.update({t.lower(): "exam" for t in other_tests})
    return sources


def build_step_state(trajectory, step: int, case_id: str, sources: dict[str, str], test_list: list[str]) -> StepState:
    """Snapshot what the agents can see before the Decision Agent's next action."""
    evidence = []
    for test in trajectory.requested_tests:
        text = trajectory.test_results[test]
        evidence.append(
            EvidenceItem(
                name=trajectory.test_results_lower_to_original.get(test, test),
                source=sources.get(test, "unknown"),
                available=text != NOT_AVAILABLE,
                text=text,
            )
        )
    # Leakage guard: evidence may only come from tests the agent has requested.
    requested = set(trajectory.requested_tests)
    leaked = [e.name for e in evidence if e.name.lower() not in requested]
    if leaked:
        raise AssertionError(f"Evidence for unrequested tests: {leaked}")

    hypotheses = getattr(trajectory, "hypothesis", [])
    confidences = getattr(trajectory, "confidence", [])
    action_log = getattr(trajectory, "action_log", [])
    return StepState(
        case_id=str(case_id),
        step=step,
        patient_history=trajectory.patient_history,
        evidence=evidence,
        hypothesis=hypotheses[-1] if hypotheses else None,
        confidence=confidences[-1] if confidences else None,
        previous_hypothesis=hypotheses[-2] if len(hypotheses) > 1 else None,
        request_count=trajectory.request_count,
        remaining_tests=[t for t in test_list if t.lower() not in requested],
        previous_outcome=action_log[-1]["outcome"] if action_log else None,
    )


def parse_lab_values(text: str) -> list[tuple[str, str]]:
    """Split a lab panel string ("Name: value, Name: value") into pairs."""
    return [(name.strip(), value.strip()) for name, value in _LAB_PAIR.findall(text)]


def deterministic_audit(state: StepState, cfg: dict | None = None) -> AuditResult:
    cfg = cfg or {}
    high_conf = cfg.get("high_confidence", 0.8)
    low_conf = cfg.get("low_confidence", 0.4)
    flags: list[Flag] = []

    # 1. Missingness: requested tests the oracle could not provide.
    for item in state.evidence:
        if not item.available:
            flags.append(Flag("unavailable_result", SEVERITY_WARNING, item.name,
                              f"{item.name} was requested but is not available; do not request it again"))

    # 2. Lab value quality and conflicts across panels (BMP/CMP/etc. overlap).
    seen: dict[str, tuple[str, str]] = {}
    for item in state.evidence:
        if item.source != "lab" or not item.available:
            continue
        for analyte, value in parse_lab_values(item.text):
            if value.lower() in INVALID_LAB_VALUES:
                flags.append(Flag("invalid_lab_value", SEVERITY_CRITICAL, f"{item.name}/{analyte}",
                                  f"{analyte} in {item.name} has no usable value"))
                continue
            key = analyte.lower()
            if key in seen and seen[key][1] != value:
                flags.append(Flag("lab_conflict", SEVERITY_CRITICAL, analyte,
                                  f"{analyte} differs between {seen[key][0]} ({seen[key][1]}) and {item.name} ({value})"))
            else:
                seen.setdefault(key, (item.name, value))

    # 3. Equivocal imaging / exam findings.
    for item in state.evidence:
        if item.source in ("imaging", "exam") and item.available:
            lowered = item.text.lower()
            hits = [t for t in AMBIGUITY_TERMS if t in lowered]
            if hits:
                flags.append(Flag("ambiguous_report", SEVERITY_WARNING, item.name,
                                  f"{item.name} contains hedged findings ({', '.join(hits[:3])})"))

    # 4. Process errors from the previous action.
    if state.previous_outcome in ACTION_ERROR_OUTCOMES:
        flags.append(Flag("previous_action_error", SEVERITY_CRITICAL, state.previous_outcome,
                          "the previous action was rejected by the environment; use the exact test or diagnosis names"))
    if state.request_count > len(state.evidence):
        flags.append(Flag("repeated_request", SEVERITY_CRITICAL, "tests",
                          "a test was requested more than once"))

    # 5. Hypothesis Agent state (only when the environment uses one).
    available_objective = [e for e in state.evidence if e.available and e.source in ("lab", "imaging")]
    if state.hypothesis is not None:
        if state.hypothesis == "unknown":
            flags.append(Flag("hypothesis_unknown", SEVERITY_WARNING, "hypothesis",
                              "the hypothesis agent gave no valid hypothesis"))
        if state.confidence is None:
            flags.append(Flag("confidence_missing", SEVERITY_WARNING, "hypothesis",
                              "the hypothesis agent gave no confidence"))
        if state.previous_hypothesis is not None and state.previous_hypothesis != state.hypothesis:
            flags.append(Flag("hypothesis_changed", SEVERITY_WARNING, "hypothesis",
                              f"the hypothesis changed from {state.previous_hypothesis} to {state.hypothesis}"))
        if state.confidence is not None:
            if state.confidence >= high_conf and not available_objective:
                flags.append(Flag("high_confidence_little_evidence", SEVERITY_WARNING, "hypothesis",
                                  "confidence is high but no lab or imaging result supports it yet"))
            if state.confidence <= low_conf:
                flags.append(Flag("low_confidence", SEVERITY_INFO, "hypothesis",
                                  f"hypothesis confidence is low ({state.confidence:.1f})"))

    # 6. Coverage gaps: informational, never injected on their own.
    if not any(e.available and e.source == "lab" for e in state.evidence):
        flags.append(Flag("no_lab_yet", SEVERITY_INFO, "lab", "no laboratory result available yet"))
    if not any(e.available and e.source == "imaging" for e in state.evidence):
        flags.append(Flag("no_imaging_yet", SEVERITY_INFO, "imaging", "no imaging result available yet"))

    return AuditResult(flags)
