class HABeliefReward:
    """H3b: reward the frozen HA's final probability of the correct disease.

    ``weight * p_final(y)``, where ``p_final`` is the frozen HA's last probability vector in
    the episode, scored after the last test result (``ha_probs`` from
    ``src/environment/frozen_ha_environment.py``).

    This is the episode's information gain p_final(y) - p_0(y) without p_0. p_0 is scored on
    the patient history alone, so it is the same for every rollout of a patient and cancels
    in GRPO's group normalisation (docs/HYPOTHESIS_H3.md, section 4).

    Invalid episodes (format failure, turn or length limit) get 0 when
    ``reward_invalid=False`` (the default), as they get no test cost. Otherwise failing after
    an informative test could beat a valid diagnosis without one.
    """

    __name__ = "HABeliefReward"

    def __init__(self, weight: float = 1.0, reward_invalid: bool = False) -> None:
        if weight < 0:
            raise ValueError("HA-belief weight must be nonnegative")
        self.weight = weight
        self.reward_invalid = reward_invalid

    def __call__(
        self,
        prompts: list,
        completions: list,
        condition: list[str],
        ha_probs: list[list[list[float]]] | None = None,
        ha_diseases: list | None = None,
        invalid: list[bool] | None = None,
        **kwargs,
    ) -> list[float]:
        if ha_probs is None or ha_diseases is None:
            raise ValueError("HABeliefReward needs the frozen-HA environment (ha_probs, ha_diseases)")
        if invalid is None:
            invalid = [False] * len(condition)
        rewards = []
        for label, probs, diseases, is_invalid in zip(condition, ha_probs, ha_diseases, invalid):
            if (is_invalid and not self.reward_invalid) or not probs:
                rewards.append(0.0)
                continue
            order = [d.lower() for d in diseases]
            final = probs[-1]
            if len(final) != len(order):
                raise ValueError(f"{len(final)} probabilities for {len(order)} diseases")
            rewards.append(self.weight * float(final[order.index(str(label).lower())]))
        return rewards
