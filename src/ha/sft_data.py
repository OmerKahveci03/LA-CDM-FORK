"""Supervised examples for the standalone Hypothesis Agent: full and partial patient records.

Each training patient yields several views: history only, all available tests, and
``n_random`` random subsets (size uniform in 1..n, random order). The trained Decision Agent
obtains only 1-2 results per case, so the HA must diagnose well from incomplete records;
LDTL (arXiv 2604.05116) trained its diagnosis agent on complete records only.

Pure Python: the tokenizer is passed in, so this is testable without a model.
"""

from __future__ import annotations

import random

from src.ha.prompts import ANSWER_PREFIX, DISEASES, available_tests, chat_prompt, encode, ha_user_message

IGNORE_INDEX = -100


def sample_test_subsets(
    available: list[str], rng: random.Random, n_random: int, include_none: bool = True, include_all: bool = True
) -> list[list[str]]:
    """Distinct test lists (each in the order shown to the HA) for one patient."""
    subsets: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def add(tests: list[str]) -> None:
        if tuple(tests) not in seen:
            seen.add(tuple(tests))
            subsets.append(tests)

    if include_none:
        add([])
    if include_all and available:
        add(rng.sample(available, len(available)))
    for _ in range(n_random if available else 0):
        add(rng.sample(available, rng.randint(1, len(available))))
    return subsets


def build_example(
    tokenizer, template: str, patient: dict, tests: list[str], max_tokens: int, rng: random.Random
) -> dict:
    """One training row: prompt tokens masked, target ``Hypothesis: <disease><eos>`` scored.

    Tests are dropped at random while the row is longer than ``max_tokens`` (the same 7,000-token
    budget the H1 scripted arms used for the HA prompt).
    """
    label = str(patient["condition"]).lower()
    if label not in DISEASES:
        raise ValueError(f"Unknown label {patient['condition']!r} for patient {patient['patient_id']}")
    target = encode(tokenizer, ANSWER_PREFIX) + encode(tokenizer, " " + label) + [tokenizer.eos_token_id]
    tests = list(tests)
    dropped = []
    while True:
        message = ha_user_message(template, patient["patient_history"], patient["test_results"], tests)
        prompt = encode(tokenizer, chat_prompt(tokenizer, message))
        if len(prompt) + len(target) <= max_tokens or not tests:
            break
        dropped.append(tests.pop(rng.randrange(len(tests))))
    return {
        "patient_id": patient["patient_id"],
        "label": label,
        "tests": tests,
        "dropped": dropped,
        "input_ids": prompt + target,
        "labels": [IGNORE_INDEX] * len(prompt) + target,
    }


def build_examples(
    tokenizer, template: str, patients, seed: int, n_random: int, max_tokens: int,
    include_none: bool = True, include_all: bool = True,
) -> list[dict]:
    """All views of all patients, reproducible from ``seed``."""
    rng = random.Random(seed)
    examples = []
    for patient in patients:
        available = available_tests(patient["test_results"])
        for tests in sample_test_subsets(available, rng, n_random, include_none, include_all):
            examples.append(build_example(tokenizer, template, patient, tests, max_tokens, rng))
    return examples
