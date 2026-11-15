"""Step 2: the frozen-HA environment matches the original HA turn, and FrozenHA restores model state."""

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
from src.environment.frozen_ha_environment import FrozenHAEnvironment  # noqa: E402
from src.ha.prompts import DISEASES, ha_user_message  # noqa: E402

TEMPLATE = "Diagnose.\n{provided_test_results}"
RESULTS = {"Complete Blood Count": "WBC 15\n", "CT": "inflamed pancreas\n", "MRI": "not available.\n"}
DA_TEMPLATE = "{patient_history}{first_hypothesis}"


def environment():
    return Environment(DA_TEMPLATE, 4096, 13, list(DISEASES), list(RESULTS), generate_hypothesis=True,
                       hypothesis_prompt_template=TEMPLATE, generate_confidence_calibration=True)


def trajectory(tests=("ct",)):
    t = Trajectory("x{first_hypothesis}", RESULTS, "epigastric pain")
    t.provided_tests = list(tests)
    return t


class VllmStub:
    """Answers every HA prompt with the given text, like vLLM's generate."""

    def __init__(self, text):
        self.text, self.prompts = text, []

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        self.prompts.extend(prompts)
        return [SimpleNamespace(prompt="p", prompt_token_ids=[1, 2],
                                outputs=[SimpleNamespace(text=self.text)]) for _ in prompts]


class FrozenEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.messages = []

        def scorer(message):
            self.messages.append(message)
            return [0.1, 0.1, 0.1, 0.7]  # pancreatitis, confidence 7

        self.frozen = FrozenHAEnvironment(environment(), scorer, lambda m: [len(m)], DISEASES)

    def run_both(self, old, new):
        environment()._generate_hypothesis([old], VllmStub("Hypothesis: pancreatitis\nConfidence: 7"), None, None)
        self.frozen._generate_hypothesis([new], None, None, None)

    def test_same_transcript_as_original(self):
        old, new = trajectory(), trajectory()
        self.run_both(old, new)
        self.assertEqual(new.text, old.text)
        self.assertIn("The current hypothesis is: pancreatitis (confidence: 7)", new.text)
        self.assertEqual((new.hypothesis, new.confidence), (old.hypothesis, old.confidence))
        self.assertEqual(self.messages, [ha_user_message(TEMPLATE, "epigastric pain", RESULTS, ["ct"])])
        self.assertEqual(new.ha_probs, [[0.1, 0.1, 0.1, 0.7]])

    def test_second_turn_and_error_turn_match(self):
        old, new = trajectory(), trajectory()
        self.run_both(old, new)
        old.text += "\nObservation: x\n"
        new.text += "\nObservation: x\n"
        old.error = new.error = True  # e.g. an unavailable test: keep the last hypothesis
        self.run_both(old, new)
        self.assertEqual(new.text, old.text)
        self.assertEqual(len(self.messages), 1)  # no HA call on the erroneous turn
        self.assertFalse(new.error)
        old.provided_tests.append("complete blood count")
        new.provided_tests.append("complete blood count")
        self.run_both(old, new)
        self.assertEqual(new.text, old.text)
        self.assertEqual(len(new.ha_probs), 2)

    def test_prompt_ids_and_no_calibration_pair(self):
        new = trajectory()
        self.frozen._generate_hypothesis([new], None, None, None)
        self.assertEqual(self.frozen.ha_prompt_ids, [[len(self.messages[0])]])
        self.assertIsNone(self.frozen._sample_pair([new]))
        new.request_count, new.diagnosis, new.invalid = 1, "pancreatitis", False
        kwargs = self.frozen._collect_reward_function_kwargs([new])
        self.assertEqual(kwargs[0]["ha_probs"], [[0.1, 0.1, 0.1, 0.7]])
        self.assertEqual(kwargs[0]["diagnosis"], "pancreatitis")

    def test_disease_order_checked(self):
        with self.assertRaises(ValueError):
            FrozenHAEnvironment(environment(), lambda m: [1.0], lambda m: [], ["flu"])


try:
    import peft
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:  # pragma: no cover
    peft = None


class CharTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 60 + 1 for c in text]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<u>" + messages[0]["content"] + "<a>"


@unittest.skipIf(peft is None, "peft/torch/transformers not installed")
class FrozenHATest(unittest.TestCase):
    def test_scores_with_frozen_adapter_and_restores_state(self):
        import tempfile

        from peft import LoraConfig, get_peft_model

        from src.ha.frozen import FrozenHA
        from src.ha.scoring import candidate_logprobs, softmax

        torch.manual_seed(0)
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
        lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
        with tempfile.TemporaryDirectory() as tmp:
            ha = get_peft_model(LlamaForCausalLM(config), lora)
            ha.save_pretrained(tmp)
            torch.manual_seed(1)
            model = get_peft_model(LlamaForCausalLM(config), lora)  # the "DA" adapter, trainable
            base = {k: v for k, v in ha.base_model.model.state_dict().items() if "lora_" not in k}
            model.base_model.model.load_state_dict(base, strict=False)  # same base, different LoRA
            with torch.no_grad():
                da_only = softmax(candidate_logprobs(model.eval(), [5, 6, 7], [[1], [2], [3], [4]]))
            model.train()
            frozen = FrozenHA(model, CharTokenizer(), tmp)
            got = frozen.probs("abc")
            self.assertTrue(model.training)  # restored by FrozenHA
            self.assertEqual(model.active_adapter, "default")

            ha.eval()
            expected = softmax(candidate_logprobs(ha, frozen.prompt_ids("abc"), frozen.candidates))
            self.assertNotEqual(da_only, softmax(candidate_logprobs(ha, [5, 6, 7], [[1], [2], [3], [4]])))
            model.eval()
            self.assertEqual(softmax(candidate_logprobs(model, [5, 6, 7], [[1], [2], [3], [4]])), da_only)
            model.train()
        for g, e in zip(got, expected):
            self.assertAlmostEqual(g, e, places=5)
        self.assertEqual(model.active_adapter, "default")
        self.assertTrue(model.training)
        names = {n: p.requires_grad for n, p in model.named_parameters() if "lora_" in n}
        self.assertTrue(all(v for n, v in names.items() if ".default." in n))
        self.assertFalse(any(v for n, v in names.items() if ".frozen_ha." in n))


if __name__ == "__main__":
    unittest.main()
