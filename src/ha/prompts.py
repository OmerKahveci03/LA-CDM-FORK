"""Hypothesis Agent prompts, built exactly as ``Environment._generate_hypothesis`` builds them.

Keeping the prompt identical means the new HA can later be dropped into the existing
environment, and that old and new HAs are compared on the same inputs.
``tests/test_ha_sft.py`` checks the parity against the environment.
"""

from __future__ import annotations

NOT_AVAILABLE = "not available.\n"
DISEASES = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")
# The HA answers "Hypothesis: <disease>". Scoring and training both split it as
# "Hypothesis:" + " <disease>" so the two use the same tokens.
ANSWER_PREFIX = "Hypothesis:"


def available_tests(test_results: dict[str, str]) -> list[str]:
    """Names (original case) of the tests that have a result for this patient."""
    return [name for name, result in test_results.items() if result != NOT_AVAILABLE]


def ha_user_message(template: str, patient_history: str, test_results: dict[str, str], tests: list[str]) -> str:
    """The HA's user message for a patient after ``tests`` (any case, in that order) were obtained."""
    original = {name.lower(): name for name in test_results}
    results = {name.lower(): result for name, result in test_results.items()}
    text = f"Patient History: {patient_history}\n\n"
    text += "\n\n".join(f"{original[t.lower()]}: {results[t.lower()]}" for t in tests)
    return template.format(provided_test_results=text)


def chat_prompt(tokenizer, user_message: str) -> str:
    """Chat-templated prompt text ending with the assistant header (TRL's maybe_apply_chat_template)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}], tokenize=False, add_generation_prompt=True
    )


def encode(tokenizer, text: str) -> list[int]:
    """Token ids without added special tokens (the chat template already contains them)."""
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def candidate_ids(tokenizer, diseases=DISEASES) -> list[list[int]]:
    """Token ids of " <disease>" for every disease, the part of the answer that is scored."""
    return [encode(tokenizer, " " + d) for d in diseases]
