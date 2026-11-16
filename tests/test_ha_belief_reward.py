"""H3b: the HA-belief reward reads the frozen HA's last probability of the correct disease."""

import unittest
from pathlib import Path

import yaml

from src.ha.prompts import DISEASES
from src.reward_function.ha_belief_reward import HABeliefReward

ORDER = list(DISEASES)  # appendicitis, cholecystitis, diverticulitis, pancreatitis
ROOT = Path(__file__).resolve().parents[1]


def call(reward, condition, probs, invalid=None):
    return reward(prompts=[None] * len(condition), completions=[None] * len(condition), condition=condition,
                  ha_probs=probs, ha_diseases=[ORDER] * len(condition), invalid=invalid)


class HABeliefRewardTest(unittest.TestCase):
    def test_final_probability_of_label(self):
        probs = [[[0.4, 0.3, 0.2, 0.1], [0.1, 0.1, 0.1, 0.7]]]  # history only, then after CT
        self.assertAlmostEqual(call(HABeliefReward(), ["Pancreatitis"], probs)[0], 0.7)
        self.assertAlmostEqual(call(HABeliefReward(weight=2.0), ["appendicitis"], probs)[0], 0.2)

    def test_p0_cancels_within_a_group(self):
        # Same patient, same p0: information gain and p_final give the same centred rewards.
        finals = [0.9, 0.5, 0.4, 0.4]
        probs = [[[0.4, 0.2, 0.2, 0.2], [f, (1 - f) / 3, (1 - f) / 3, (1 - f) / 3]] for f in finals]
        got = call(HABeliefReward(), ["appendicitis"] * 4, probs)
        gain = [f - 0.4 for f in finals]
        centre = lambda xs: [x - sum(xs) / len(xs) for x in xs]
        for a, b in zip(centre(got), centre(gain)):
            self.assertAlmostEqual(a, b)

    def test_invalid_and_empty_get_zero(self):
        probs = [[[0.0, 0.0, 0.0, 1.0]], []]
        self.assertEqual(call(HABeliefReward(), ["pancreatitis", "pancreatitis"], probs, [True, False]), [0.0, 0.0])
        self.assertEqual(call(HABeliefReward(reward_invalid=True), ["pancreatitis"], probs[:1], [True]), [1.0])

    def test_needs_frozen_environment_kwargs(self):
        with self.assertRaises(ValueError):
            HABeliefReward()(prompts=[None], completions=[None], condition=["appendicitis"])
        with self.assertRaises(ValueError):
            HABeliefReward(weight=-1)

    def test_h3b_config_is_defaults_plus_belief(self):
        load = lambda name: yaml.safe_load((ROOT / "configs/reward_function" / name).read_text())
        default, h3b = load("defaults.yaml"), load("h3b.yaml")
        self.assertEqual(h3b["reward_functions"][:3], default["reward_functions"])
        self.assertEqual(h3b["reward_functions"][3]["_target_"],
                         "src.reward_function.ha_belief_reward.HABeliefReward")
        self.assertEqual(h3b["conf_cal_reward_func"], default["conf_cal_reward_func"])


if __name__ == "__main__":
    unittest.main()
