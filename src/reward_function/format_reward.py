class FormatReward:
    """Reward function that penalises invalid action formatting.

    Returns ``valid_reward`` when the trajectory ended with a correctly
    formatted action, and ``invalid_reward`` otherwise.
    """

    __name__ = "FormatReward"

    def __init__(self, invalid_reward: float, valid_reward: float) -> None:
        self.invalid_reward = invalid_reward
        self.valid_reward = valid_reward

    def __call__(
        self,
        prompts: list,
        completions: list,
        invalid: list[bool],
        **kwargs,
    ) -> list[float]:
        """Compute format rewards for a batch of episodes.

        Args:
            prompts: Prompt strings (unused, required by trainer interface).
            completions: Completion strings (unused, required by trainer interface).
            invalid: Whether each trajectory ended with an invalid action.

        Returns:
            List of scalar rewards, one per episode.
        """
        rewards = [self.invalid_reward if inv else self.valid_reward for inv in invalid]
        return rewards
