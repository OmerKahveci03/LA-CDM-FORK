"""H3c: the DA's diagnosis is replaced by the frozen HA's current hypothesis."""

import sys
import types
import unittest

if "trl" not in sys.modules:
    trl = types.ModuleType("trl")
    data_utils = types.ModuleType("trl.data_utils")
    data_utils.maybe_apply_chat_template = lambda example, tokenizer: example
    trl.data_utils = data_utils
    sys.modules["trl"] = trl
    sys.modules["trl.data_utils"] = data_utils

from src.environment.defer_ha_environment import DeferHAEnvironment  # noqa: E402
from src.ha.prompts import DISEASES  # noqa: E402
from tests.test_frozen_ha import environment, trajectory  # noqa: E402


class DeferTest(unittest.TestCase):
    def setUp(self):
        self.env = DeferHAEnvironment(environment(), lambda m: [0.1, 0.1, 0.1, 0.7], lambda m: [len(m)], DISEASES)

    def test_diagnosis_becomes_the_ha_hypothesis(self):
        t = trajectory()
        self.env._generate_hypothesis([t], None, None, None)  # HA: pancreatitis
        t.diagnosis = "appendicitis"  # what the DA wrote
        t.actions.append("diagnosis:appendicitis")
        DeferHAEnvironment.apply_defer(t)
        self.assertEqual(t.diagnosis, "pancreatitis")
        self.assertEqual(t.actions[-2:], ["diagnosis:pancreatitis", "da_choice:appendicitis"])

    def test_agreement_is_logged_too(self):
        t = trajectory()
        self.env._generate_hypothesis([t], None, None, None)
        t.diagnosis = "pancreatitis"
        t.actions.append("diagnosis:pancreatitis")
        DeferHAEnvironment.apply_defer(t)
        self.assertEqual(t.diagnosis, "pancreatitis")
        self.assertEqual(t.actions[-1], "da_choice:pancreatitis")

    def test_no_ha_answer_keeps_the_da(self):
        t = trajectory()
        t.diagnosis = "appendicitis"
        t.actions.append("diagnosis:appendicitis")
        DeferHAEnvironment.apply_defer(t)
        self.assertEqual(t.diagnosis, "appendicitis")

    def test_step_applies_defer_only_when_a_diagnosis_is_made(self):
        env = self.env
        calls = []

        def fake_parent_step(self_, trajectory_, step, tokenizer):
            calls.append(step)
            if step == 2:  # the DA diagnoses at its second turn
                trajectory_.diagnosis = "appendicitis"
                trajectory_.actions.append("diagnosis:appendicitis")

        t = trajectory()
        env._generate_hypothesis([t], None, None, None)
        parent = DeferHAEnvironment.__mro__[1]  # FrozenHAEnvironment (inherits step from Environment)
        self.assertNotIn("step", vars(parent))
        parent.step = fake_parent_step
        try:
            env.step(t, 1, None)
            self.assertIsNone(t.diagnosis)
            env.step(t, 2, None)
        finally:
            del parent.step
        self.assertEqual(t.diagnosis, "pancreatitis")
        self.assertEqual(calls, [1, 2])


if __name__ == "__main__":
    unittest.main()
