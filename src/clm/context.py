"""Pure-Python parts of the CLM Hypothesis Agent (no torch, unit-tested in tests/test_clm.py).

The HA's context starts as ``Patient History: ...`` and each test result is appended in
exactly the format of ``src.ha.prompts.ha_user_message``. So an HA that never rewrites sees
the same text as the H2 HA, and scoring it reproduces the H2 scores. At any turn the HA may
reply ``<rewrite>...</rewrite>`` to replace its whole context with notes; later test results
are appended to those notes.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from src.ha.prompts import DISEASES, available_tests, ha_user_message

REWRITE = re.compile(r"<rewrite>\s*(.*?)\s*</rewrite>", re.DOTALL | re.IGNORECASE)
KEEP, REWROTE, MALFORMED, TOO_LONG, APPEND_ONLY = "keep", "rewrite", "malformed", "too_long", "append_only"


def initial_context(patient_history: str) -> str:
    return f"Patient History: {patient_history}\n\n"


def test_block(name: str, result: str) -> str:
    return f"{name}: {result}"


def parse_completion(text: str) -> tuple[str, str | None]:
    """(KEEP | REWROTE | MALFORMED, new context or None) for one reply of the HA."""
    match = REWRITE.search(text)
    if match:
        notes = match.group(1).strip()
        return (REWROTE, notes) if notes else (MALFORMED, None)
    if "<rewrite>" in text.lower():
        return MALFORMED, None  # opened but never closed (e.g. ran out of tokens)
    return KEEP, None


@dataclass
class CLMState:
    """One HA context. ``observe`` appends a test result; ``apply`` handles the HA's reply."""

    context: str
    needs_separator: bool = False
    seen_tests: int = 0  # environment use: how many provided tests are already in the context
    rewrites: int = 0

    @classmethod
    def start(cls, patient_history: str) -> "CLMState":
        return cls(initial_context(patient_history))

    def observe(self, block: str) -> None:
        self.context += ("\n\n" if self.needs_separator else "") + block
        self.needs_separator = True

    def apply(self, completion_text: str) -> str:
        status, notes = parse_completion(completion_text)
        if status == REWROTE:
            self.context = notes
            self.needs_separator = True
            self.rewrites += 1
        return status


def blocks_for(patient: dict, tests: list[str]) -> list[str]:
    """Test-result blocks (original test names) for ``tests`` in the given order."""
    original = {name.lower(): name for name in patient["test_results"]}
    results = {name.lower(): result for name, result in patient["test_results"].items()}
    return [test_block(original[t.lower()], results[t.lower()]) for t in tests]


def append_only_context(patient: dict, tests: list[str]) -> str:
    """The context of an HA that never rewrites, after ``tests`` (equals ha_user_message's text)."""
    state = CLMState.start(patient["patient_history"])
    for block in blocks_for(patient, tests):
        state.observe(block)
    return state.context


def fit_tests(count_tokens, template: str, patient: dict, tests: list[str], max_tokens: int,
              rng: random.Random) -> list[str]:
    """Drop tests at random until the append-only scoring prompt fits ``max_tokens``.

    ``count_tokens(message) -> int`` counts the chat-templated prompt. Same rule as the H2 data
    (src/ha/sft_data.py), so long records are cut the same way.
    """
    tests = list(tests)
    while tests:
        message = ha_user_message(template, patient["patient_history"], patient["test_results"], tests)
        if count_tokens(message) <= max_tokens:
            break
        tests.pop(rng.randrange(len(tests)))
    return tests


def random_sequence(patient: dict, rng: random.Random, p_all: float = 0.5) -> list[str]:
    """Training view: all available tests (probability ``p_all``) or a random non-empty subset,
    in random order. Patients without any test result get an empty sequence (history only)."""
    available = available_tests(patient["test_results"])
    if not available:
        return []
    size = len(available) if rng.random() < p_all else rng.randint(1, len(available))
    return rng.sample(available, size)


# ------------------------------------------------------------------- advantages

def outcome_advantages(rewards: list[float], eps: float = 1e-4) -> list[float]:
    """Standard GRPO advantage within one group: (r - mean) / (std + eps); zero if all equal."""
    n = len(rewards)
    mean = sum(rewards) / n
    std = (sum((r - mean) ** 2 for r in rewards) / n) ** 0.5
    if std == 0:
        return [0.0] * n
    return [(r - mean) / (std + eps) for r in rewards]


def success_gated_efficiency(costs: list[float], success: list[bool]) -> list[float]:
    """The CLM paper's Eq. 6: only successful rollouts are ranked by cost.

    A_eff = clip((mean cost of successes - c_i) / mean cost of successes, -1, 1) for a success,
    0 for a failure, and 0 for everyone when fewer than two rollouts succeed.
    """
    winners = [c for c, s in zip(costs, success) if s]
    if len(winners) < 2:
        return [0.0] * len(costs)
    mean = sum(winners) / len(winners)
    if mean <= 0:
        return [0.0] * len(costs)
    return [max(-1.0, min(1.0, (mean - c) / mean)) if s else 0.0 for c, s in zip(costs, success)]


def combined_advantages(rewards: list[float], costs: list[float], success: list[bool], w_eff: float) -> list[float]:
    """A_i = A_out_i + w_eff * A_eff_i (the paper's combination)."""
    out = outcome_advantages(rewards)
    eff = success_gated_efficiency(costs, success)
    return [a + w_eff * e for a, e in zip(out, eff)]


def top_disease(probs: list[float], diseases=DISEASES) -> str:
    return diseases[max(range(len(probs)), key=probs.__getitem__)]


@dataclass
class TurnRecord:
    """What the HA did at one turn (token ids are kept for training, dropped from logs)."""

    action: str
    context: str
    probs: list[float]
    context_tokens: int
    score_prompt_ids: list[int] = field(default_factory=list, repr=False)
    edit_prompt_ids: list[int] = field(default_factory=list, repr=False)
    completion_ids: list[int] = field(default_factory=list, repr=False)
    completion: str = ""

    def log(self) -> dict:
        return {"action": self.action, "context_tokens": self.context_tokens,
                "probs": [round(p, 4) for p in self.probs], "context": self.context,
                "completion": self.completion}
