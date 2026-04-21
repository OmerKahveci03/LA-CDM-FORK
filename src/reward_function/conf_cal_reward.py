import math

class ConfidenceCalibrationReward:
    """Reward function for the Hypothesis Agent's confidence calibration.

    Scores a predicted confidence value (0–10) using a log-scoring rule that
    encourages well-calibrated confidence. Out-of-format predictions receive
    ``oof_reward``.
    """

    __name__ = "ConfidenceCalibrationReward"

    def __init__(
        self,
        oof_reward: float = -2.0,
        reward_scale: float = 1.0,
        correctness_bonus: float = 0.0,
    ) -> None:
        self.oof_reward = oof_reward
        self.reward_scale = reward_scale
        self.correctness_bonus = correctness_bonus

    def __call__(
        self,
        completions: list[str],
        hypothesis: str,
        label: str,
        **kwargs,
    ) -> list[float]:
        """Compute confidence calibration rewards.

        Args:
            completions: Predicted confidence strings (expected ``"0"``–``"10"``).
            hypothesis: The hypothesis being evaluated.
            label: Ground-truth condition.

        Returns:
            List of scalar rewards, one per completion.
        """
        rewards = []
        for conf in completions:            
            # Compute reward
            if conf not in [str(i) for i in range(0, 11)]:
                score = self.oof_reward
            else:
                correct = hypothesis == label
                conf = int(conf) / 10
                score = max(conf, 0.001) if correct else max(1 - conf, 0.001)
                score = self.reward_scale * math.log(score)
                score = -1 + ((score - math.log(0.001)) / (math.log(1) - math.log(0.001))) * 2
                if correct:
                    score += self.correctness_bonus
            rewards.append(score)
        return rewards
