"""Deterministic activation gate for the adaptive Evidence Critic.

Thresholds are configuration, to be tuned on the validation split only.
"""

from src.trace_cdm.schemas import SEVERITY_CRITICAL, AuditResult, GateDecision, StepState


def gate(state: StepState, audit: AuditResult, mode: str, calls_so_far: int, cfg: dict | None = None) -> GateDecision:
    """Decide whether to call the critic at this decision point.

    Args:
        audit: Flags that are new in this episode, so an unchanged finding does
            not re-trigger the critic at every later step.
        mode: ``"none"``, ``"adaptive"`` or ``"always"``.
        calls_so_far: Critic calls already made in this episode.
    """
    cfg = cfg or {}
    if mode == "none":
        return GateDecision(False, [])
    max_calls = cfg.get("max_calls_per_episode")
    if max_calls is not None and calls_so_far >= max_calls:
        return GateDecision(False, ["episode_call_budget_exhausted"])
    if mode == "always":
        return GateDecision(True, ["always_on"])
    if mode != "adaptive":
        raise ValueError(f"Unknown critic mode: {mode}")

    reasons = []
    if audit.of_severity(SEVERITY_CRITICAL):
        reasons.append("critical_evidence_flag")
    if audit.has("high_confidence_little_evidence"):
        reasons.append("confidence_evidence_disagreement")
    if audit.has("ambiguous_report") and state.confidence is not None \
            and state.confidence >= cfg.get("ambiguous_high_confidence", 0.8):
        reasons.append("confident_on_equivocal_evidence")
    if audit.has("hypothesis_changed"):
        reasons.append("hypothesis_instability")
    if audit.has("low_confidence") and state.evidence:
        # Low confidence at step 0 is expected; only trigger once tests are in.
        reasons.append("low_confidence_after_testing")
    if audit.has("hypothesis_unknown") or audit.has("confidence_missing"):
        reasons.append("hypothesis_agent_failure")
    return GateDecision(bool(reasons), reasons)
