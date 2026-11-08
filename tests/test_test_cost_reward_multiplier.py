import unittest

from src.reward_function.test_cost_reward import TestCostReward


class TestCostRewardMultiplier(unittest.TestCase):
    def test_scales_sum_of_individual_test_costs(self):
        reward = TestCostReward(test_cost={"ct": -0.2, "cbc": -0.01}, single_cost=False, multiplier=1.5)
        values = reward(prompts=[], completions=[], provided_tests=[["ct", "cbc"], []])
        self.assertAlmostEqual(values[0], -0.315)
        self.assertEqual(values[1], 0)

    def test_zero_multiplier_removes_penalty(self):
        reward = TestCostReward(test_cost=-0.1, single_cost=True, multiplier=0)
        self.assertEqual(reward(prompts=[], completions=[], provided_tests=[["ct", "cbc"]]), [0])

    def test_negative_multiplier_is_rejected(self):
        with self.assertRaises(ValueError):
            TestCostReward(multiplier=-1)


class TestCostRewardInvalidEpisodes(unittest.TestCase):
    COSTS = {"ct": -0.2, "cbc": -0.01}

    def test_default_charges_invalid_episodes(self):
        reward = TestCostReward(test_cost=self.COSTS, single_cost=False)
        values = reward(prompts=[], completions=[], provided_tests=[["ct"], ["ct"]], invalid=[False, True])
        self.assertEqual(values, [-0.2, -0.2])

    def test_gated_reward_does_not_charge_invalid_episodes(self):
        reward = TestCostReward(test_cost=self.COSTS, single_cost=False, charge_invalid=False)
        values = reward(prompts=[], completions=[], provided_tests=[["ct", "cbc"], ["ct"]], invalid=[False, True])
        self.assertAlmostEqual(values[0], -0.21)
        self.assertEqual(values[1], 0.0)

    def test_all_failed_group_ties(self):
        # The collapse case: every rollout fails format, one before any test.
        # Gated, all four get the same reward, so GRPO gives zero advantage
        # instead of rewarding the rollout that failed first.
        reward = TestCostReward(test_cost=self.COSTS, single_cost=False, charge_invalid=False)
        values = reward(prompts=[], completions=[], provided_tests=[[], ["cbc"], ["ct"], ["ct", "cbc"]],
                        invalid=[True] * 4)
        self.assertEqual(len(set(values)), 1)

    def test_gating_requires_invalid_flags(self):
        reward = TestCostReward(test_cost=self.COSTS, single_cost=False, charge_invalid=False)
        with self.assertRaises(ValueError):
            reward(prompts=[], completions=[], provided_tests=[["ct"]])


if __name__ == "__main__":
    unittest.main()
