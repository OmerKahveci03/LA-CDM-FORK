from dataclasses import dataclass
import logging


@dataclass
class Phase:
    """A single phase in the loss schedule."""

    name: str
    steps: int
    learning_rate: float


class LossScheduler:
    """Cycles loss weights through training phases for the three objectives.

    The LA-CDM training schedule alternates between Decision Agent GRPO,
    Hypothesis Agent SFT, and Hypothesis Agent confidence calibration GRPO.
    Each phase activates exactly one loss objective for a fixed number of
    training steps, optionally repeating the full cycle.
    """

    def __init__(self, phases: list[dict], repeat: bool = True) -> None:
        """Initialize the loss scheduler.

        Args:
            phases: Phase configs, each with ``name``, ``steps``, and
                ``learning_rate`` keys.
            repeat: Whether to loop after the last phase.
        """
        self.phases = [Phase(**phase) for phase in phases]
        self.repeat = repeat
        self.current_phase_idx = 0
        self.steps_in_current_phase = 0
        self.total_steps = sum(phase.steps for phase in self.phases)
        self.current_step = 0
        
        # Validate phase names
        valid_names = {"da", "ha_sft", "ha_conf_cal"}
        for phase in self.phases:
            if phase.name not in valid_names:
                raise ValueError(f"Invalid phase name: {phase.name}. Must be one of {valid_names}")
        
        logging.info(f"Initialized loss scheduler with {len(self.phases)} phases")
        for phase in self.phases:
            logging.info(f"Phase: {phase.name}, Steps: {phase.steps}")

    def step(self) -> tuple[dict[str, float], float | None]:
        """Advance one step and return updated loss weights and learning rate.

        Returns:
            Tuple of ``(weights_dict, learning_rate)``.
        """
        self.current_step += 1
        self.steps_in_current_phase += 1
        
        # Check if we need to move to the next phase
        if self.steps_in_current_phase >= self.phases[self.current_phase_idx].steps:
            self.current_phase_idx = (self.current_phase_idx + 1) % len(self.phases)
            self.steps_in_current_phase = 0
            
            # If we've completed all phases and repeat is False, keep the last phase
            if not self.repeat and self.current_phase_idx == 0:
                self.current_phase_idx = len(self.phases) - 1
                self.steps_in_current_phase = self.phases[-1].steps
        
        # Get current phase
        current_phase = self.phases[self.current_phase_idx]
        
        # Set weights based on current phase
        weights = {
            "da_loss_weight": 1.0 if current_phase.name == "da" else 0.0,
            "ha_sft_loss_weight": 1.0 if current_phase.name == "ha_sft" else 0.0,
            "ha_conf_cal_loss_weight": 1.0 if current_phase.name == "ha_conf_cal" else 0.0
        }
        learning_rate = current_phase.learning_rate
        
        return weights, learning_rate


    def get_progress(self) -> float:
        """Return overall schedule progress as a float in ``[0, 1]``."""
        if not self.repeat:
            return min(1.0, self.current_step / self.total_steps)
        return (self.current_step % self.total_steps) / self.total_steps 