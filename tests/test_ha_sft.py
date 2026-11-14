"""Standalone Hypothesis Agent (H2): prompt parity, training rows, scoring and loss."""

import random
import sys
import types
import unittest
from types import SimpleNamespace

if "trl" not in sys.modules:
    trl = types.ModuleType("trl")
    data_utils = types.ModuleType("trl.data_utils")
    data_utils.maybe_apply_chat_template = lambda example, tokenizer: example
    trl.data_utils = data_utils
    sys.modules["trl"] = trl
    sys.modules["trl.data_utils"] = data_utils

from src.environment.environment import Environment, Trajectory  # noqa: E402
from src.ha.prompts import DISEASES, available_tests, ha_user_message  # noqa: E402
from src.ha.records import probability_summary, scored_row  # noqa: E402
from src.ha.sft_data import IGNORE_INDEX, build_example, build_examples, sample_test_subsets  # noqa: E402

NA = "not available.\n"
TEMPLATE = "Diagnose.\n{provided_test_results}"
RESULTS = {"Complete Blood Count": "WBC 15\n", "CT": "inflamed appendix\n", "MRI": NA, "Urinalysis": "normal\n"}


class CharTokenizer:
    """One token per character; the chat template wraps the user message in markers."""

    eos_token, eos_token_id = "</s>", 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 97 + 1 for c in text]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<u>" + messages[0]["content"] + "<a>"


def patient(pid=1, label="Appendicitis", results=RESULTS):
    return {"patient_id": pid, "condition": label, "patient_history": "RLQ pain", "test_results": dict(results)}


class PromptParityTest(unittest.TestCase):
    def test_matches_environment(self):
        captured = []

        class Model:
            def generate(self, prompts, sampling_params=None, use_tqdm=False):
                captured.extend(prompts)
                return [SimpleNamespace(prompt="p", prompt_token_ids=[1],
                                        outputs=[SimpleNamespace(text="Hypothesis: appendicitis\nConfidence: 7")])
                        for _ in prompts]

        env = Environment("{patient_history}{first_hypothesis}", 4096, 13, list(DISEASES), list(RESULTS),
                          generate_hypothesis=True, hypothesis_prompt_template=TEMPLATE,
                          generate_confidence_calibration=True)
        trajectory = Trajectory("x{first_hypothesis}", RESULTS, "RLQ pain")
        trajectory.provided_tests = ["ct", "complete blood count"]  # request order, lower case
        env._generate_hypothesis([trajectory], Model(), None, None)
        expected = ha_user_message(TEMPLATE, "RLQ pain", RESULTS, ["ct", "complete blood count"])
        self.assertEqual(captured[0][0]["content"], expected)
        self.assertIn("CT: inflamed appendix\n\n\nComplete Blood Count: WBC 15\n", expected)

    def test_history_only(self):
        self.assertEqual(ha_user_message(TEMPLATE, "RLQ pain", RESULTS, []), "Diagnose.\nPatient History: RLQ pain\n\n")


class SftDataTest(unittest.TestCase):
    def test_available_tests(self):
        self.assertEqual(available_tests(RESULTS), ["Complete Blood Count", "CT", "Urinalysis"])

    def test_subsets(self):
        available = available_tests(RESULTS)
        subsets = sample_test_subsets(available, random.Random(0), n_random=5)
        self.assertEqual(subsets[0], [])
        self.assertEqual(sorted(subsets[1]), sorted(available))
        self.assertEqual(len({tuple(s) for s in subsets}), len(subsets))  # no duplicates
        for s in subsets:
            self.assertTrue(set(s) <= set(available))
        self.assertEqual(subsets, sample_test_subsets(available, random.Random(0), n_random=5))  # reproducible
        self.assertEqual(sample_test_subsets([], random.Random(0), n_random=3), [[]])

    def test_example_masks_prompt(self):
        tok = CharTokenizer()
        e = build_example(tok, TEMPLATE, patient(), ["CT"], 10_000, random.Random(0))
        target = tok("Hypothesis:")["input_ids"] + tok(" appendicitis")["input_ids"] + [0]
        self.assertEqual(e["input_ids"][-len(target):], target)
        self.assertEqual(e["labels"][-len(target):], target)
        self.assertTrue(all(x == IGNORE_INDEX for x in e["labels"][:-len(target)]))
        self.assertEqual(len(e["labels"]), len(e["input_ids"]))
        self.assertEqual(e["label"], "appendicitis")

    def test_budget_drops_tests(self):
        tok = CharTokenizer()
        full = build_example(tok, TEMPLATE, patient(), ["CT", "Complete Blood Count"], 10_000, random.Random(0))
        short = build_example(tok, TEMPLATE, patient(), ["CT", "Complete Blood Count"], len(full["input_ids"]) - 1,
                              random.Random(0))
        self.assertEqual(len(short["tests"]), 1)
        self.assertEqual(len(short["dropped"]), 1)

    def test_unknown_label(self):
        with self.assertRaises(ValueError):
            build_example(CharTokenizer(), TEMPLATE, patient(label="gastritis"), [], 10_000, random.Random(0))

    def test_build_examples(self):
        rows = build_examples(CharTokenizer(), TEMPLATE, [patient(1), patient(2, results={"CT": NA})], 0, 2, 10_000)
        self.assertEqual([len(r["tests"]) for r in rows if r["patient_id"] == 2], [0])
        self.assertGreaterEqual(sum(r["patient_id"] == 1 for r in rows), 2)


class RecordsTest(unittest.TestCase):
    def test_scored_row(self):
        source = {"patient_id": "7", "label": "pancreatitis", "tests": ["ct"], "cost_usd": 1306.0}
        row = scored_row(source, [0.1, 0.1, 0.1, 0.7])
        self.assertEqual((row["diagnosis"], row["correct"], row["final_confidence"]), ("pancreatitis", True, 7))
        self.assertEqual((row["num_tests"], row["cost_usd"]), (1, 1306.0))

    def test_probability_summary(self):
        labels = ["appendicitis", "cholecystitis"]
        probs = [[0.9, 0.1, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0]]
        s = probability_summary(labels, probs)
        self.assertEqual(s["accuracy"], 0.5)
        self.assertEqual(s["auroc_per_disease"]["appendicitis"], 1.0)
        self.assertEqual(s["auroc_per_disease"]["diverticulitis"], None)
        self.assertAlmostEqual(s["brier"], ((0.1 ** 2 + 0.1 ** 2) + (0.6 ** 2 + 0.6 ** 2)) / 2)


try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch/transformers not installed")
class TorchTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
        self.model = LlamaForCausalLM(config).eval()

    def test_candidate_logprobs_match_full_pass(self):
        from src.ha.scoring import candidate_logprobs, softmax

        prompt, candidates = [5, 9, 12, 3, 7], [[20], [21, 22, 23], [24, 25], [26, 27, 28, 29]]
        got = candidate_logprobs(self.model, prompt, candidates)
        for cand, value in zip(candidates, got):
            with torch.no_grad():
                logp = torch.log_softmax(self.model(torch.tensor([prompt + cand])).logits[0].float(), -1)
            expected = sum(logp[len(prompt) - 1 + j, t].item() for j, t in enumerate(cand))
            self.assertAlmostEqual(value, expected, places=4)
        self.assertAlmostEqual(sum(softmax(got)), 1.0)

    def test_suffix_nll_matches_masked_loss(self):
        from src.ha.scoring import suffix_nll

        ids = torch.tensor([[5, 9, 12, 3, 7, 30, 31, 0]])
        labels = torch.tensor([[-100, -100, -100, -100, -100, 30, 31, 0]])
        mask = torch.ones_like(ids)
        with torch.no_grad():
            expected = self.model(input_ids=ids, attention_mask=mask, labels=labels).loss
            got = suffix_nll(self.model, ids, mask, labels)
        self.assertAlmostEqual(got.item(), expected.item(), places=5)


if __name__ == "__main__":
    unittest.main()
