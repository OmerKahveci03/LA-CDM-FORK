"""Fixed test-ordering policies in place of the Decision Agent (hypothesis H1).

Everything except the Decision Agent's turn is the normal environment: the trained
Hypothesis Agent reads the patient history and each returned test result, with the
same prompt, sampling settings and adapter as a regular test evaluation. The
scripted policy orders its tests one per turn and then diagnoses with the Hypothesis
Agent's latest hypothesis. The output files have the usual format, so
``summarize_episodes.py``, ``test_stats.py`` and ``plot_run.py`` read them unchanged.

Policies:

- ``none``: diagnose from the patient history alone.
- ``all``: order every test that has a result for this patient, in the order of
  ``test_list``. Tests without a result are skipped, because they would return
  "not available" and the Hypothesis Agent would learn nothing from them.
- ``random_matched``: for each patient, order the same number of tests that the
  trained Decision Agent obtained results for in a reference test run
  (``num_tests`` in its ``test_episodes.csv``), chosen at random among the tests
  available for that patient. Only *which* tests are ordered differs from the
  reference, not how many.

If the Hypothesis Agent has no valid hypothesis when the policy is ready to
diagnose, the episode ends as ``invalid_format`` (counted as undiagnosed).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from types import SimpleNamespace

from src.environment.environment import Environment, EnvironmentOutput, Trajectory
from src.utils.episode_log import read_rows

POLICIES = ("none", "all", "random_matched")
NOT_AVAILABLE = "not available.\n"


class ScriptedDecisionEnvironment(Environment):
    def __init__(
        self,
        *args,
        policy: str,
        reference_episodes: str | None = None,
        policy_seed: int = 0,
        ha_token_budget: int = 7000,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if policy not in POLICIES:
            raise ValueError(f"Unknown scripted policy {policy!r}; choose from {POLICIES}")
        if not self.generate_hypothesis:
            raise ValueError("Scripted policies diagnose with the Hypothesis Agent; enable generate_hypothesis")
        self.policy = policy
        self.policy_seed = policy_seed
        # The scripted Decision Agent's own transcript is never sent to the model,
        # so its length limit does not apply. The Hypothesis Agent's prompt is
        # bounded by ha_token_budget instead (vLLM context: 8,192 tokens).
        self.max_length = 10**9
        self.ha_token_budget = ha_token_budget
        self.reference_counts: dict[str, int] = {}
        if policy == "random_matched":
            if not reference_episodes:
                raise ValueError("random_matched needs reference_episodes (a test_episodes.csv)")
            rows = read_rows(Path(reference_episodes))
            self.reference_counts = {str(r["patient_id"]): int(r["num_tests"]) for r in rows}
        self._plans: dict[int, list[str]] = {}

    # ------------------------------------------------------------------ plan

    def _available_tests(self, trajectory: Trajectory) -> list[str]:
        return [t for t in self.test_list if trajectory.test_results.get(t.lower(), NOT_AVAILABLE) != NOT_AVAILABLE]

    def _plan(self, trajectory: Trajectory, tokenizer) -> list[str]:
        available = self._available_tests(trajectory)
        if self.policy == "none":
            tests = []
        elif self.policy == "all":
            tests = available
        else:
            patient = str(trajectory.patient_id)
            if patient not in self.reference_counts:
                raise KeyError(f"Patient {patient} is not in the reference episodes")
            k = min(self.reference_counts[patient], len(available))
            tests = random.Random(f"{self.policy_seed}:{patient}").sample(available, k)
        return self._within_budget(trajectory, tests, tokenizer)

    def _within_budget(self, trajectory: Trajectory, tests: list[str], tokenizer) -> list[str]:
        """Drop the tests that would push the Hypothesis Agent's prompt past the budget."""
        used = len(tokenizer.encode(f"Patient History: {trajectory.patient_history}\n\n", add_special_tokens=False))
        kept = []
        for test in tests:
            cost = len(tokenizer.encode(f"{test}: {trajectory.test_results[test.lower()]}\n\n", add_special_tokens=False))
            if used + cost > self.ha_token_budget:
                logging.warning("Patient %s: skipping %s to keep the HA prompt under %d tokens",
                                trajectory.patient_id, test, self.ha_token_budget)
                continue
            used += cost
            kept.append(test)
        return kept

    # ------------------------------------------------------------------ act

    def _decide(self, trajectories, prompts, model, sampling_params):
        tokenizer = model.get_tokenizer()
        outputs = []
        for trajectory, prompt in zip(trajectories, prompts):
            if id(trajectory) not in self._plans:
                self._plans[id(trajectory)] = self._plan(trajectory, tokenizer)
            plan = self._plans[id(trajectory)]
            if plan:
                text = f"Thought: Scripted {self.policy} policy.\nAction: Test\nAction Input: {plan.pop(0)}"
                ids = tokenizer.encode(text, add_special_tokens=False)
            else:
                hypothesis = trajectory.hypothesis[-1] if trajectory.hypothesis else "unknown"
                if hypothesis in [d.lower() for d in self.disease_list]:
                    text = f"Thought: Scripted {self.policy} policy.\nAction: Diagnosis\nAction Input: {hypothesis}"
                else:
                    text = "No valid hypothesis."  # parsed as invalid_format: undiagnosed
                ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
            outputs.append(SimpleNamespace(
                prompt=prompt,
                prompt_token_ids=tokenizer.encode(prompt, add_special_tokens=False),
                outputs=[SimpleNamespace(token_ids=ids, text=text)],
            ))
        return outputs

    def run(self, prompts_text, inputs, model, sampling_params, processing_class) -> EnvironmentOutput:
        self._plans.clear()
        output = super().run(prompts_text, inputs, model, sampling_params, processing_class)
        # The trainer scores each Decision Agent transcript with the model even in
        # evaluation. The scripted transcripts are not model outputs, so replace
        # them with the last action only: short, and with matching mask lengths.
        tokenizer = model.get_tokenizer()
        short = [
            tokenizer.encode(record["last_completion"], add_special_tokens=False) or [tokenizer.eos_token_id]
            for record in output.episode_records
        ]
        output.completion_ids = short
        output.completion_mask = [[True] * len(ids) for ids in short]
        return output
