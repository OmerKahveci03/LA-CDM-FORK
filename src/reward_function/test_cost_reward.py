class TestCostReward:
    """Reward function that penalises the number (or cost) of ordered tests.

    When ``single_cost=True``, every test incurs the same flat penalty.
    When ``False``, ``test_cost`` should be a dict mapping test names to
    individual costs.

    When ``charge_invalid=False``, episodes that ended invalid (format failure,
    turn or length limit) get no cost penalty. Otherwise an episode that fails
    before ordering any test is the cheapest rollout in its GRPO group, and
    groups in which every rollout fails reward failing early. That signal
    collapsed the 1.5x arm and the November 2026 full-pass 1x run. The paper's
    reward charges every episode (``charge_invalid=True``, the default here).
    """

    __name__ = "TestCostReward"

    def __init__(
        self,
        test_cost: float | dict[str, float] = -0.1,
        single_cost: bool = True,
        multiplier: float = 1.0,
        charge_invalid: bool = True,
    ) -> None:
        if multiplier < 0:
            raise ValueError("test-cost multiplier must be nonnegative")
        self.test_cost = test_cost
        self.single_cost = single_cost
        self.multiplier = multiplier
        self.charge_invalid = charge_invalid

    def __call__(
        self,
        prompts: list,
        completions: list,
        provided_tests: list[list[str]],
        invalid: list[bool] | None = None,
        **kwargs,
    ) -> list[float]:
        if invalid is None:
            if not self.charge_invalid:
                raise ValueError("charge_invalid=False needs the per-episode 'invalid' flags")
            invalid = [False] * len(provided_tests)
        rewards = []
        for test_list, is_invalid in zip(provided_tests, invalid):
            if is_invalid and not self.charge_invalid:
                rewards.append(0.0)
                continue
            if self.single_cost:
                reward = self.test_cost * len(test_list)
            else:
                reward = sum(self.test_cost[test] for test in test_list)
            rewards.append(self.multiplier * reward)
        return rewards
