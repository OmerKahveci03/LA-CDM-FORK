"""The frozen-HA environment where the DA never overrides the HA (H3c, docs/HYPOTHESIS_H3C.md).

The DA still decides which tests to order and when to stop, and it still writes
"Action: Diagnosis / Action Input: <disease>" to stop (so format and the action space are unchanged).
But the diagnosis that counts (the one stored on the trajectory, rewarded and logged) is always
the frozen HA's current hypothesis. Its own choice is kept in the episode's actions as
``da_choice:<disease>`` for analysis. Training therefore rewards the DA for gathering the tests that
make the *HA's* answer right.

Everything else is ``FrozenHAEnvironment``: the HA scores every turn, and its message is shown to the
DA as before.
"""

from __future__ import annotations

from src.environment.environment import Trajectory
from src.environment.frozen_ha_environment import FrozenHAEnvironment


class DeferHAEnvironment(FrozenHAEnvironment):
    def step(self, trajectory: Trajectory, step: int, tokenizer) -> None:
        before = trajectory.diagnosis
        super().step(trajectory, step, tokenizer)
        if before is None and trajectory.diagnosis is not None:
            self.apply_defer(trajectory)

    @staticmethod
    def apply_defer(trajectory: Trajectory) -> None:
        """Replace the DA's diagnosis with the HA's latest hypothesis (if the HA has given one)."""
        hypothesis = trajectory.hypothesis[-1] if trajectory.hypothesis else None
        if not hypothesis or hypothesis == "unknown":
            return  # no HA answer to defer to: keep the DA's
        da_choice = trajectory.diagnosis
        trajectory.diagnosis = hypothesis
        if trajectory.actions and trajectory.actions[-1] == f"diagnosis:{da_choice}":
            trajectory.actions[-1] = f"diagnosis:{hypothesis}"
        trajectory.actions.append(f"da_choice:{da_choice}")
