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


if __name__ == "__main__":
    unittest.main()
