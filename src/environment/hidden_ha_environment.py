"""The frozen-HA environment with the HA hidden from the DA (ablation A4a, docs/FINAL_ANALYSES.md).

The HA is still asked at every turn and logged exactly as in ``FrozenHAEnvironment`` (so the episode
files keep ``hypotheses``, ``final_hypothesis`` and ``final_confidence``), but the line
"The current hypothesis is: ..." is never added to the DA's transcript. The DA's instructions are
unchanged. This measures how much a DA trained *with* an HA relies on it; it is not a DA trained
without one.
"""

from __future__ import annotations

from src.environment.environment import Trajectory
from src.environment.frozen_ha_environment import FrozenHAEnvironment

PLACEHOLDER = "{first_hypothesis}"


class HiddenHAEnvironment(FrozenHAEnvironment):
    def _generate_hypothesis(self, trajectories: list[Trajectory], model, sampling_params, processing_class) -> list[str]:
        before = [t.text for t in trajectories]
        super()._generate_hypothesis(trajectories, model, sampling_params, processing_class)
        for trajectory, text in zip(trajectories, before):
            # Undo the transcript change only; the HA's records stay on the trajectory.
            trajectory.text = text.replace(PLACEHOLDER, "") if PLACEHOLDER in text else text
        return [trajectory.text for trajectory in trajectories]
