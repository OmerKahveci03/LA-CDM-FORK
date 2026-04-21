class DiagnosisReward:
    """Reward function that scores a diagnosis against the ground-truth condition.

    Returns ``correct_reward`` for an exact match, ``incorrect_reward`` for a
    wrong diagnosis, and ``undiagnosed_reward`` when no diagnosis was made.
    """

    __name__ = "DiagnosisReward"

    def __init__(
        self,
        correct_reward: float = 1.0,
        incorrect_reward: float = -1.0,
        undiagnosed_reward: float = 0.0,
    ) -> None:
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.undiagnosed_reward = undiagnosed_reward

    def __call__(
        self,
        prompts: list,
        completions: list,
        condition: list[str],
        diagnosis: list[str | None],
        **kwargs,
    ) -> list[float]:
        """Compute rewards for a batch of episodes.

        Args:
            prompts: Prompt strings (unused, required by trainer interface).
            completions: Completion strings (unused, required by trainer interface).
            condition: Ground-truth diagnoses.
            diagnosis: Predicted diagnoses (``None`` if undiagnosed).

        Returns:
            List of scalar rewards, one per episode.
        """
        rewards = []
        for diagnosis, true_condition in zip(diagnosis, condition):            
            # Compute reward
            if diagnosis is None:
                reward = self.undiagnosed_reward
            elif diagnosis.lower() == true_condition.lower():
                reward = self.correct_reward
            else:
                reward = self.incorrect_reward
            rewards.append(reward)
            
        return rewards
