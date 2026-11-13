"""Scripted Decision Agent policies for hypothesis H1, with a fake model (no GPU)."""

import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

if "trl" not in sys.modules:
    trl = types.ModuleType("trl")
    data_utils = types.ModuleType("trl.data_utils")
    data_utils.maybe_apply_chat_template = lambda example, tokenizer: example
    trl.data_utils = data_utils
    sys.modules["trl"] = trl
    sys.modules["trl.data_utils"] = data_utils

from src.environment.scripted_environment import ScriptedDecisionEnvironment  # noqa: E402

DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]
TESTS = ["Complete Blood Count", "Liver Function Panel", "CT", "Ultrasound", "Physical Examination"]
NA = "not available.\n"


class CharTokenizer:
    eos_token, eos_token_id = "</s>", 0

    def encode(self, text, add_special_tokens=False):
        return [1] * len(str(text))

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [self.encode(t) for t in texts]}


class FakeModel:
    """Every generation is a Hypothesis Agent answer (the scripted policy never calls generate)."""

    def __init__(self, hypothesis="pancreatitis"):
        self.tokenizer, self.hypothesis, self.ha_prompts = CharTokenizer(), hypothesis, []

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        self.ha_prompts.extend(prompts)
        text = f"Hypothesis: {self.hypothesis}\nConfidence: 7"
        # The real chat template returns a string; the trl stand-in returns the messages.
        return [SimpleNamespace(prompt=str(p), prompt_token_ids=[1] * 5, outputs=[SimpleNamespace(text=text, token_ids=[1, 1])])
                for p in prompts]


def patient(pid, available):
    return {
        "patient_id": pid,
        "condition": "pancreatitis",
        "patient_history": "epigastric pain",
        "test_results": {t: (f"{t} result\n" if t in available else NA) for t in TESTS},
    }


def make_env(policy, **kwargs):
    return ScriptedDecisionEnvironment(
        prompt_template="{patient_history}",
        max_length=2048,
        max_steps=13,
        disease_list=DISEASES,
        test_list=TESTS,
        generate_hypothesis=True,
        hypothesis_prompt_template="{provided_test_results}",
        generate_confidence_calibration=True,
        policy=policy,
        **kwargs,
    )


def run(env, inputs, model=None):
    model = model or FakeModel()
    tok = model.get_tokenizer()
    return env.run(["prompt"] * len(inputs), inputs, model, None, tok), model


def write_reference(path, counts):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "num_tests"])
        writer.writeheader()
        for pid, n in counts.items():
            writer.writerow({"patient_id": pid, "num_tests": n})


class ScriptedPolicyTest(unittest.TestCase):
    def test_none_diagnoses_from_history(self):
        out, model = run(make_env("none"), [patient(1, {"CT", "Ultrasound"})])
        record = out.episode_records[0]
        self.assertEqual(record["provided_tests"], [])
        self.assertEqual(record["diagnosis"], "pancreatitis")
        self.assertEqual(record["end_reason"], "diagnosis")
        self.assertEqual(record["num_turns"], 1)

    def test_all_orders_every_available_test_once(self):
        available = {"Complete Blood Count", "CT", "Physical Examination"}
        out, model = run(make_env("all"), [patient(1, available)])
        record = out.episode_records[0]
        self.assertEqual(record["provided_tests"], ["complete blood count", "ct", "physical examination"])
        self.assertEqual(record["num_unavailable_requests"], 0)
        self.assertEqual(record["diagnosis"], "pancreatitis")
        self.assertEqual(record["num_turns"], 4)
        # The last HA call saw every result.
        self.assertIn("Physical Examination result", str(model.ha_prompts[-2]))

    def test_random_matched_uses_reference_count_and_is_repeatable(self):
        available = {"Complete Blood Count", "Liver Function Panel", "CT", "Ultrasound"}
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "test_episodes.csv"
            write_reference(ref, {1: 2, 2: 0, 3: 9})
            inputs = [patient(1, available), patient(2, available), patient(3, {"CT"})]
            first, _ = run(make_env("random_matched", reference_episodes=str(ref)), inputs)
            second, _ = run(make_env("random_matched", reference_episodes=str(ref)), inputs)
            other_seed, _ = run(make_env("random_matched", reference_episodes=str(ref), policy_seed=1), inputs)
        tests = [r["provided_tests"] for r in first.episode_records]
        self.assertEqual(len(tests[0]), 2)
        self.assertTrue({t.lower() for t in available} >= set(tests[0]))
        self.assertEqual(tests[1], [])
        self.assertEqual(tests[2], ["ct"])  # capped at what is available
        self.assertEqual(tests, [r["provided_tests"] for r in second.episode_records])
        self.assertEqual(len(other_seed.episode_records[0]["provided_tests"]), 2)

    def test_random_matched_rejects_unknown_patient(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "test_episodes.csv"
            write_reference(ref, {1: 1})
            with self.assertRaises(KeyError):
                run(make_env("random_matched", reference_episodes=str(ref)), [patient(2, {"CT"})])

    def test_no_valid_hypothesis_is_undiagnosed(self):
        out, _ = run(make_env("none"), [patient(1, {"CT"})], FakeModel(hypothesis="gastritis"))
        record = out.episode_records[0]
        self.assertIsNone(record["diagnosis"])
        self.assertEqual(record["end_reason"], "invalid_format")

    def test_completion_ids_and_mask_align(self):
        out, _ = run(make_env("all"), [patient(1, {"CT", "Ultrasound"}), patient(2, set())])
        for ids, mask in zip(out.completion_ids, out.completion_mask):
            self.assertGreater(len(ids), 0)
            self.assertEqual(len(ids), len(mask))

    def test_budget_drops_tests_that_do_not_fit(self):
        env = make_env("all", ha_token_budget=len("Patient History: epigastric pain\n\n") + len("CT: CT result\n\n\n"))
        out, _ = run(env, [patient(1, {"CT", "Ultrasound"})])
        self.assertEqual(out.episode_records[0]["provided_tests"], ["ct"])

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            make_env("smart")


if __name__ == "__main__":
    unittest.main()
