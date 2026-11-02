class TestCostReward:
    """Reward function that penalises the number (or cost) of ordered tests.

    When ``single_cost=True``, every test incurs the same flat penalty.
    When ``False``, ``test_cost`` should be a dict mapping test names to
    individual costs.
    """

    __name__ = "TestCostReward"

    def __init__(self, test_cost: float | dict[str, float] = -0.1, single_cost: bool = True, multiplier: float = 1.0) -> None:
        if multiplier < 0:
            raise ValueError("test-cost multiplier must be nonnegative")
        self.test_cost = test_cost
        self.single_cost = single_cost
        self.multiplier = multiplier

    def __call__(self, prompts: list, completions: list, provided_tests: list[list[str]], **kwargs) -> list[float]:
        rewards = []
        if self.single_cost:
            for test_list in provided_tests:
                reward = self.test_cost * len(test_list)
                rewards.append(self.multiplier * reward)
        else:
            for test_list in provided_tests:
                reward = 0
                for test in test_list:
                    reward += self.test_cost[test]
                rewards.append(self.multiplier * reward)
        return rewards
