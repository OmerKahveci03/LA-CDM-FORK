"""The normal environment, but a frozen standalone HA gives the hypotheses (step 2, H3).

Only the Hypothesis Agent's turn changes. ``Environment._generate_hypothesis`` asks the
vLLM model (the shared LA-CDM adapter) to write ``Hypothesis: ...\\nConfidence: ...``. Here
the frozen HA scores the four diseases on the same prompt instead (``src/ha/frozen.py``):

- the hypothesis is the most probable disease;
- the confidence is round(10 x its probability), on the same 0-10 scale the DA was
  trained to read.

The text added to the DA's transcript has the same format as before, so the trained DA
reads the new HA's output exactly as it read the old one.

The frozen HA is never trained. The confidence-calibration pair is therefore not sampled:
``_sample_pair`` returns None, so no calibration prompts are generated. Everything else
(the DA's turns, actions, parsing, rewards and episode records) is inherited unchanged.
The full probability vector of every HA call is kept in ``trajectory.ha_probs`` and passed to
the reward functions as ``ha_probs``. The existing rewards ignore it; H3's
information-gain reward will use it.
"""

from __future__ import annotations

from typing import Callable

from src.environment.environment import Environment, Trajectory
from src.ha.prompts import ha_user_message


class FrozenHAEnvironment(Environment):
    """Wraps an already configured ``Environment`` and replaces its HA turn.

    ``scorer(user_message) -> probabilities`` in ``disease_order``. ``prompt_ids(user_message)``
    returns the token ids the scorer used; the trainer's HA tensors are built from them.
    """

    def __init__(self, base: Environment, scorer: Callable[[str], list[float]],
                 prompt_ids: Callable[[str], list[int]], disease_order) -> None:
        self.__dict__.update(base.__dict__)
        if not self.generate_hypothesis:
            raise ValueError("The frozen HA replaces the HA turn; enable generate_hypothesis")
        self.scorer = scorer
        self.prompt_ids_of = prompt_ids
        self.disease_order = [d.lower() for d in disease_order]
        if sorted(self.disease_order) != sorted(d.lower() for d in self.disease_list):
            raise ValueError(f"Scorer diseases {self.disease_order} do not match {self.disease_list}")

    def _generate_hypothesis(self, trajectories: list[Trajectory], model, sampling_params, processing_class) -> list[str]:
        prompt_ids = []
        for trajectory in trajectories:
            if not hasattr(trajectory, "ha_probs"):
                trajectory.ha_probs = []
            if trajectory.error:
                # As in the parent: an erroneous turn (unknown or unavailable test) keeps the last
                # hypothesis. The parent then prints the stored 0-1 value (e.g. "0.7"), not the
                # 0-10 one; the DA was trained on that text, so it is kept.
                hypothesis = trajectory.hypothesis[-1] if trajectory.hypothesis else "unknown"
                confidence = trajectory.confidence[-1] if trajectory.confidence else None
                trajectory.error = False
            else:
                # Trajectory keys are lower case; the prompt uses the original names ("CT", not "ct").
                original = {trajectory.test_results_lower_to_original[k]: v for k, v in trajectory.test_results.items()}
                message = ha_user_message(self.hypothesis_prompt_template, trajectory.patient_history,
                                          original, trajectory.provided_tests)
                probs = self.scorer(message)
                top = max(range(len(probs)), key=probs.__getitem__)
                hypothesis = self.disease_order[top]
                confidence = round(10 * probs[top])
                trajectory.ha_probs.append(list(probs))
                trajectory.hypothesis.append(hypothesis)
                trajectory.hypothesis_prompt.append(message)
                if self.generate_confidence_calibration:
                    trajectory.confidence.append(confidence / 10)
                prompt_ids.append(self.prompt_ids_of(message))

            # Same transcript text as Environment._generate_hypothesis.
            if self.generate_confidence_calibration:
                suffix = f"The current hypothesis is: {hypothesis.lower()}"
                suffix += f" (confidence: {confidence})" if confidence is not None and hypothesis != "unknown" else ""
                suffix += "\n"
            else:
                suffix = f"The current hypothesis is: {hypothesis.lower()}\n"
            if "{first_hypothesis}" in trajectory.text:
                trajectory.text = trajectory.text.format(first_hypothesis=f"\n\n{suffix}")
            else:
                trajectory.text += suffix
        self.ha_prompt_ids.extend(prompt_ids)
        return [trajectory.text for trajectory in trajectories]

    def _sample_pair(self, trajectories, forbidden_value: str = "unknown"):
        return None  # the frozen HA is not calibrated during training

    def _collect_reward_function_kwargs(self, trajectories: list[Trajectory]) -> list[dict]:
        kwargs = super()._collect_reward_function_kwargs(trajectories)
        for kw, trajectory in zip(kwargs, trajectories):
            kw["ha_probs"] = list(getattr(trajectory, "ha_probs", []))
            kw["ha_diseases"] = list(self.disease_order)  # the order of each ha_probs vector
        return kwargs
