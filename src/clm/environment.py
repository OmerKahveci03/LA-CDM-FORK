"""The normal environment with the CLM HA in the HA turn (H4 stage 4).

Like ``src/environment/frozen_ha_environment.py`` (which is unchanged), only the HA turn
differs. Each trajectory keeps its own ``CLMState``; at every HA turn the test results
obtained since the last turn are appended, the HA may rewrite its context, and the four-way
scores give the hypothesis and the confidence. The text added to the DA's transcript has the
same format as before ("The current hypothesis is: X (confidence: c)").

During training (model in train mode) the HA samples its replies and each finished trajectory
is put in ``self.buffer`` for the HA update (src/clm/joint.py). During validation and test it
replies greedily. With ``notes_log_path`` set, every trajectory's HA turns are written there.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.clm.agent import CLMHA
from src.clm.context import CLMState, test_block
from src.environment.environment import Environment, Trajectory


class CLMHAEnvironment(Environment):
    def __init__(self, base: Environment, agent: CLMHA, notes_log_path: str | None = None) -> None:
        self.__dict__.update(base.__dict__)
        if not self.generate_hypothesis:
            raise ValueError("The CLM HA replaces the HA turn; enable generate_hypothesis")
        self.agent = agent
        self.disease_order = [d.lower() for d in agent.diseases]
        if sorted(self.disease_order) != sorted(d.lower() for d in self.disease_list):
            raise ValueError(f"HA diseases {self.disease_order} do not match {self.disease_list}")
        self.notes_log_path = Path(notes_log_path) if notes_log_path else None
        self.buffer: list[dict] = []
        self._run_id = 0
        self._label = None
        self._patient_id = None
        self._sample = False

    def run(self, prompts_text, inputs, model, sampling_params, processing_class):
        self._run_id += 1
        self._label = str(inputs[0]["condition"]).lower()
        self._patient_id = inputs[0].get("patient_id")
        self._sample = bool(self.agent.model.training)  # Trainer.evaluate() puts the model in eval mode
        return super().run(prompts_text, inputs, model, sampling_params, processing_class)

    @staticmethod
    def _state(trajectory: Trajectory) -> CLMState:
        if not hasattr(trajectory, "clm_state"):
            trajectory.clm_state = CLMState.start(trajectory.patient_history)
            trajectory.clm_turns = []
            trajectory.ha_probs = []
        return trajectory.clm_state

    def _generate_hypothesis(self, trajectories, model, sampling_params, processing_class) -> list[str]:
        live = [t for t in trajectories if not t.error]
        states = []
        for t in live:
            state = self._state(t)
            for name in t.provided_tests[state.seen_tests:]:
                state.observe(test_block(t.test_results_lower_to_original[name], t.test_results[name]))
            state.seen_tests = len(t.provided_tests)
            states.append(state)
        records = dict(zip(map(id, live), self.agent.turn(states, sample=self._sample))) if live else {}

        prompt_ids = []
        for t in trajectories:
            self._state(t)
            if t.error:
                # As in the original and frozen-HA environments: keep the last hypothesis.
                hypothesis = t.hypothesis[-1] if t.hypothesis else "unknown"
                confidence = t.confidence[-1] if t.confidence else None
                t.error = False
            else:
                record = records[id(t)]
                top = max(range(len(record.probs)), key=record.probs.__getitem__)
                hypothesis = self.disease_order[top]
                confidence = round(10 * record.probs[top])
                t.ha_probs.append(list(record.probs))
                t.clm_turns.append(record)
                t.hypothesis.append(hypothesis)
                t.hypothesis_prompt.append(record.context)
                if self.generate_confidence_calibration:
                    t.confidence.append(confidence / 10)
                prompt_ids.append(record.score_prompt_ids)
            if self.generate_confidence_calibration:
                suffix = f"The current hypothesis is: {hypothesis.lower()}"
                suffix += f" (confidence: {confidence})" if confidence is not None and hypothesis != "unknown" else ""
                suffix += "\n"
            else:
                suffix = f"The current hypothesis is: {hypothesis.lower()}\n"
            if "{first_hypothesis}" in t.text:
                t.text = t.text.format(first_hypothesis=f"\n\n{suffix}")
            else:
                t.text += suffix
        self.ha_prompt_ids.extend(prompt_ids)
        return [t.text for t in trajectories]

    def _sample_pair(self, trajectories, forbidden_value: str = "unknown"):
        return None  # no confidence-calibration prompts: the confidence is read from probabilities

    def _collect_reward_function_kwargs(self, trajectories: list[Trajectory]) -> list[dict]:
        kwargs = super()._collect_reward_function_kwargs(trajectories)
        for kw, t in zip(kwargs, trajectories):
            kw["ha_probs"] = list(getattr(t, "ha_probs", []))
            kw["ha_diseases"] = list(self.disease_order)
        # End of the episode group: hand the HA's turns to the update and the log.
        for t in trajectories:
            turns = getattr(t, "clm_turns", [])
            if not turns:
                continue
            if self._sample:
                self.buffer.append({"run": self._run_id, "label": self._label, "turns": turns})
            if self.notes_log_path is not None:
                self.notes_log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.notes_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"mode": "train" if self._sample else "eval",
                                             "patient_id": self._patient_id, "label": self._label,
                                             "tests": list(t.provided_tests), "diagnosis": t.diagnosis,
                                             "turns": [r.log() for r in turns]}) + "\n")
        return kwargs

    def take_groups(self) -> list[list[dict]]:
        """Empty the buffer into groups (the rollouts of one patient = one environment run)."""
        groups: dict[int, list[dict]] = {}
        for entry in self.buffer:
            groups.setdefault(entry["run"], []).append(entry)
        self.buffer = []
        return list(groups.values())
