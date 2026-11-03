"""Typed contracts shared by the audit, gate, critic and controller."""

from dataclasses import dataclass, field, asdict

NOT_AVAILABLE = "not available.\n"

SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

CRITIC_NEXT_STEPS = ("proceed", "request_test", "review")


@dataclass
class EvidenceItem:
    """One test the agent has already requested at this decision point."""

    name: str  # display name, e.g. "Complete Blood Count"
    source: str  # "lab", "imaging" or "exam"
    available: bool  # False when the oracle answered "not available"
    text: str  # result text exactly as shown to the Decision Agent


@dataclass
class StepState:
    """Everything visible to the agents before the Decision Agent acts.

    Built only from the trajectory's own record of requested tests, never from
    the patient's full ``test_results`` table or the label.
    """

    case_id: str
    step: int
    patient_history: str
    evidence: list[EvidenceItem]
    hypothesis: str | None  # None when the environment has no Hypothesis Agent
    confidence: float | None  # 0-1, Hypothesis Agent's stated confidence / 10
    previous_hypothesis: str | None
    request_count: int
    remaining_tests: list[str]  # tests not yet requested, the valid next requests
    previous_outcome: str | None  # environment outcome of the previous action


@dataclass
class Flag:
    type: str
    severity: str
    item: str
    message: str


@dataclass
class AuditResult:
    flags: list[Flag] = field(default_factory=list)

    def of_severity(self, *severities: str) -> list[Flag]:
        return [f for f in self.flags if f.severity in severities]

    def has(self, flag_type: str) -> bool:
        return any(f.type == flag_type for f in self.flags)

    def to_dict(self) -> dict:
        return {"flags": [asdict(f) for f in self.flags]}


@dataclass
class GateDecision:
    invoke: bool
    reasons: list[str]


@dataclass
class CriticResult:
    evidence_concerns: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    recommended_next_step: str = "review"
    recommended_test: str | None = None
    reason: str = ""
    # Uncalibrated self-reported score in [0, 1]; never treat as a probability.
    evidence_quality: float | None = None
    parse_error: str | None = None
    invalid_recommendation: bool = False
    shuffled_from: str | None = None  # case_id of the donor in the shuffled arm

    def to_dict(self) -> dict:
        return asdict(self)
