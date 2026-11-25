"""A4a: the hidden-HA environment logs the HA like the frozen one but never shows it to the DA."""

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

from src.environment.frozen_ha_environment import FrozenHAEnvironment  # noqa: E402
from src.environment.hidden_ha_environment import HiddenHAEnvironment  # noqa: E402
from src.ha.prompts import DISEASES  # noqa: E402
from tests.test_frozen_ha import environment, trajectory  # noqa: E402


def scorer(message):
    return [0.1, 0.1, 0.1, 0.7]  # pancreatitis, confidence 7


class HiddenEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.shown = FrozenHAEnvironment(environment(), scorer, lambda m: [len(m)], DISEASES)
        self.hidden = HiddenHAEnvironment(environment(), scorer, lambda m: [len(m)], DISEASES)

    def test_first_turn_hides_text_but_logs_ha(self):
        shown, hidden = trajectory(), trajectory()
        self.shown._generate_hypothesis([shown], None, None, None)
        self.hidden._generate_hypothesis([hidden], None, None, None)
        self.assertIn("The current hypothesis is", shown.text)
        self.assertEqual(hidden.text, "x")  # the placeholder is removed, nothing added
        self.assertEqual((hidden.hypothesis, hidden.confidence, hidden.ha_probs),
                         (shown.hypothesis, shown.confidence, shown.ha_probs))

    def test_later_and_error_turns_add_nothing(self):
        hidden = trajectory()
        self.hidden._generate_hypothesis([hidden], None, None, None)
        hidden.text += "\nObservation: x\n"
        before = hidden.text
        hidden.provided_tests.append("complete blood count")
        self.hidden._generate_hypothesis([hidden], None, None, None)
        self.assertEqual(hidden.text, before)
        hidden.error = True
        self.hidden._generate_hypothesis([hidden], None, None, None)
        self.assertEqual(hidden.text, before)
        # The erroneous turn reuses the last hypothesis without a new HA call or record.
        self.assertEqual(hidden.hypothesis, ["pancreatitis", "pancreatitis"])
        self.assertEqual(len(hidden.ha_probs), 2)
        self.assertFalse(hidden.error)

    def test_entry_point_swaps_the_class(self):
        import importlib

        try:
            importlib.import_module("hydra")
        except ImportError:
            self.skipTest("hydra not installed")
        module = importlib.import_module("src.evaluate_hidden_ha")
        self.assertIs(sys.modules["src.evaluate_frozen_ha"].FrozenHAEnvironment, HiddenHAEnvironment)
        self.assertTrue(callable(module.main))


if __name__ == "__main__":
    unittest.main()
