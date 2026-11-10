"""HA-SFT loss: every right-padded row must contribute (regression test for the
September 2026 suffix-scoring bug), and the result must equal the paper's original
computation (full forward with masked labels). Needs torch and transformers."""

import importlib.util
import unittest

HAS_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers"))

PAD, PLACEHOLDER = 0, 7


@unittest.skipUnless(HAS_TORCH, "needs torch and transformers")
class HaSftLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        cls.torch = torch
        cls.model = LlamaForCausalLM(LlamaConfig(
            vocab_size=40, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
            pad_token_id=PAD)).eval()
        # Same target suffix for every row ("Hypothesis: X / Confidence: 0 <eos>" stand-in),
        # prompts of different lengths, right-padded like TRL's pad().
        label = [21, 22, 23, PLACEHOLDER, 24]
        prompts = [[3, 4, 5], [3, 4, 5, 6, 8, 9, 10], [3, 4, 5, 11, 12]]
        rows = [p + label for p in prompts]
        width = max(map(len, rows))
        cls.ids = torch.tensor([r + [PAD] * (width - len(r)) for r in rows])
        cls.label_mask = torch.tensor([[0] * len(p) + [1] * len(label) + [0] * (width - len(p) - len(label))
                                       for p in prompts])
        cls.attention = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])

    def token_logps(self, ids, attention, keep):
        """Same slicing as CDMGRPOTrainer._get_per_token_logps."""
        logits = self.model(input_ids=ids, attention_mask=attention, logits_to_keep=keep + 1).logits[:, :-1, :]
        logits = logits[:, -keep:]
        return logits.log_softmax(-1).gather(-1, ids[:, -keep:].unsqueeze(-1)).squeeze(-1)

    def paper_reference(self):
        """Original LA-CDM code: per-row full forward with masked labels, token-weighted."""
        torch = self.torch
        scored = self.label_mask.bool() & (self.ids != PLACEHOLDER)
        total, loss = scored.sum(), torch.zeros(())
        for i in range(self.ids.size(0)):
            labels = self.ids[i : i + 1].clone()
            labels[~scored[i : i + 1]] = -100
            out = self.model(input_ids=self.ids[i : i + 1], attention_mask=self.attention[i : i + 1], labels=labels)
            loss = loss + out.loss * scored[i].sum() / total
        return loss

    def test_matches_paper_computation(self):
        from src.trainer.ha_sft_loss import ha_sft_loss

        with self.torch.no_grad():
            ours = ha_sft_loss(self.ids, self.label_mask, self.attention, PLACEHOLDER, self.token_logps)
            reference = self.paper_reference()
        self.assertAlmostEqual(ours.item(), reference.item(), places=5)

    def test_every_row_contributes(self):
        from src.trainer.ha_sft_loss import ha_sft_loss

        seen = []

        def recording(ids, attention, keep):
            seen.append(ids[0, -keep:].tolist())
            return self.token_logps(ids, attention, keep)

        with self.torch.no_grad():
            ha_sft_loss(self.ids, self.label_mask, self.attention, PLACEHOLDER, recording)
        # Each of the 3 rows is scored, and what is scored is the label, never padding.
        self.assertEqual(seen, [[21, 22, 23, PLACEHOLDER, 24]] * 3)

    def test_old_suffix_scoring_dropped_shorter_rows(self):
        """Documents the bug: scoring the padded row's last k positions reads padding."""
        width = self.ids.size(1)
        for i in range(self.ids.size(0)):
            keep = int(self.label_mask[i].sum())
            window = self.label_mask[i, width - keep:]
            is_longest = int(self.attention[i].sum()) == width
            self.assertEqual(bool(window.all()), is_longest)

    def test_rejects_left_padding(self):
        from src.trainer.ha_sft_loss import ha_sft_loss

        torch = self.torch
        ids = torch.tensor([[PAD, 3, 21, 24]])
        with self.assertRaises(ValueError):
            ha_sft_loss(ids, torch.tensor([[0, 0, 1, 1]]), torch.tensor([[0, 1, 1, 1]]), PLACEHOLDER,
                        self.token_logps)


if __name__ == "__main__":
    unittest.main()
