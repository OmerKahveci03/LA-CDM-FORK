"""The resume fix (src/utils/peft_resume.py) on a tiny two-adapter Llama, saved like a checkpoint."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch


def tiny_two_adapter_model(base):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], lora_dropout=0.0)
    model = get_peft_model(copy.deepcopy(base), config)  # "default" = the DA
    model.add_adapter("frozen_ha", config)  # the second, frozen adapter
    model.set_adapter("default")
    return model


class PeftResumeTest(unittest.TestCase):
    def setUp(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        self.base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                                 num_hidden_layers=1, num_attention_heads=2,
                                                 num_key_value_heads=2))

    def lora(self, model, adapter):
        return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n and f".{adapter}." in n}

    def test_checkpoint_layout_and_restore(self):
        from src.utils.peft_resume import RestoreAdapterCallback, adapter_norm, load_root_adapter

        trained = tiny_two_adapter_model(self.base)
        with torch.no_grad():  # "train" both adapters so they differ from a fresh init
            for name, param in trained.named_parameters():
                if "lora_" in name:
                    param.add_(torch.randn_like(param))
        with tempfile.TemporaryDirectory() as tmp:
            trained.save_pretrained(tmp)
            # The layout that trips transformers' resume: DA at the root, the other adapter in a subfolder.
            self.assertTrue((Path(tmp) / "adapter_model.safetensors").is_file())
            self.assertTrue((Path(tmp) / "frozen_ha" / "adapter_model.safetensors").is_file())

            fresh = tiny_two_adapter_model(self.base)  # what the resumed job builds
            self.assertNotAlmostEqual(adapter_norm(fresh), adapter_norm(trained), places=3)
            params_before = {n: p for n, p in fresh.named_parameters()}

            info = load_root_adapter(fresh, tmp)
            self.assertGreater(info["tensors"], 0)
            want, got = self.lora(trained, "default"), self.lora(fresh, "default")
            self.assertEqual(want.keys(), got.keys())
            for name in want:
                torch.testing.assert_close(got[name], want[name])
            # In place: the optimizer's parameter objects are still the model's parameters.
            for name, param in fresh.named_parameters():
                self.assertIs(param, params_before[name])
            # The frozen adapter is untouched (still the fresh one).
            for name, value in self.lora(fresh, "frozen_ha").items():
                self.assertFalse(torch.equal(value, self.lora(trained, "frozen_ha")[name]))

            again = tiny_two_adapter_model(self.base)
            RestoreAdapterCallback(tmp).on_train_begin(None, None, None, model=again)
            torch.testing.assert_close(adapter_norm(again), adapter_norm(trained))
            RestoreAdapterCallback(None).on_train_begin(None, None, None, model=again)  # no resume: no-op


if __name__ == "__main__":
    unittest.main()
