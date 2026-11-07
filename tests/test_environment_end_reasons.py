import sys
import types
import unittest

# The environment imports one helper from TRL. Provide a stand-in so these
# tests run without the training stack installed.
if "trl" not in sys.modules:
    trl = types.ModuleType("trl")
    data_utils = types.ModuleType("trl.data_utils")
    data_utils.maybe_apply_chat_template = lambda example, tokenizer: example
    trl.data_utils = data_utils
    sys.modules["trl"] = trl
    sys.modules["trl.data_utils"] = data_utils

from src.environment.environment import Environment, Trajectory  # noqa: E402


class CharTokenizer:
    """One token per character keeps length arithmetic obvious."""

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


TESTS = ["CT", "Ultrasound", "Complete Blood Count"]


def make_env(max_steps=13, max_length=10_000):
    return Environment(
        prompt_template="{patient_history}",
        max_length=max_length,
        max_steps=max_steps,
        disease_list=["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"],
        test_list=TESTS,
    )


def make_trajectory():
    results = {"CT": "appendix dilated\n", "Ultrasound": "not available.\n", "Complete Blood Count": "WBC 14\n"}
    return Trajectory("prompt", results, "history")


def act(env, trajectory, completion, step):
    trajectory.last_completion = completion
    trajectory.token_mask += [True] * len(completion)
    env.step(trajectory, step, CharTokenizer())


def action(kind, value):
    return f"Thought: reasoning\nAction: {kind}\nAction Input: {value}"


class EnvironmentEndReasonTest(unittest.TestCase):
    def test_full_episode_records_every_action(self):
        env, trajectory = make_env(), make_trajectory()
        act(env, trajectory, action("Test", "CT"), 0)
        act(env, trajectory, action("Test", "Ultrasound"), 1)
        act(env, trajectory, action("Test", "Complete Blood Count (CBC)"), 2)
        act(env, trajectory, action("Test", "CT"), 3)
        act(env, trajectory, action("Diagnosis", "appendicitis"), 4)

        self.assertTrue(trajectory.completed)
        self.assertFalse(trajectory.invalid)
        self.assertEqual(trajectory.end_reason, "diagnosis")
        self.assertEqual(trajectory.num_turns, 5)
        self.assertEqual(
            trajectory.actions,
            ["test:ct", "unavailable:ultrasound", "unknown_test:complete blood count (cbc)",
             "repeat:ct", "diagnosis:appendicitis"],
        )
        self.assertEqual(trajectory.provided_tests, ["ct"])
        self.assertEqual(trajectory.num_unavailable_requests, 1)
        self.assertEqual(trajectory.num_unknown_test_requests, 1)
        self.assertEqual(trajectory.num_repeat_requests, 1)

    def test_wrong_but_valid_diagnosis_ends_episode(self):
        env, trajectory = make_env(), make_trajectory()
        act(env, trajectory, action("Diagnosis", "pancreatitis"), 0)
        self.assertEqual(trajectory.end_reason, "diagnosis")
        self.assertEqual(trajectory.diagnosis, "pancreatitis")

    def test_unknown_diagnosis_continues(self):
        env, trajectory = make_env(), make_trajectory()
        act(env, trajectory, action("Diagnosis", "gallstones"), 0)
        self.assertFalse(trajectory.completed)
        self.assertEqual(trajectory.num_unknown_diagnoses, 1)

    def test_unparseable_output_is_invalid_format(self):
        env, trajectory = make_env(), make_trajectory()
        act(env, trajectory, "Thought: I need more information.", 0)
        self.assertTrue(trajectory.invalid)
        self.assertEqual(trajectory.end_reason, "invalid_format")

    def test_turn_limit(self):
        env, trajectory = make_env(max_steps=1), make_trajectory()
        act(env, trajectory, action("Test", "CT"), 0)
        act(env, trajectory, action("Test", "Complete Blood Count"), 1)
        self.assertFalse(trajectory.completed)
        act(env, trajectory, action("Test", "CT"), 2)
        self.assertEqual(trajectory.end_reason, "max_steps")

    def test_length_limit_is_not_reported_as_format(self):
        env, trajectory = make_env(max_length=20), make_trajectory()
        act(env, trajectory, action("Diagnosis", "appendicitis"), 0)
        self.assertTrue(trajectory.invalid)
        self.assertEqual(trajectory.end_reason, "max_length")

    def test_observation_overflow_is_length_limit(self):
        completion = action("Test", "CT")
        env, trajectory = make_env(max_length=len(completion) + 5), make_trajectory()
        act(env, trajectory, completion, 0)
        self.assertEqual(trajectory.end_reason, "max_length")


if __name__ == "__main__":
    unittest.main()
