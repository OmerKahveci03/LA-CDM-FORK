"""H4: the contrastive HA (src/contrastive_ha/). Needs torch and transformers (tiny Llama, CPU)."""

import json
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

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:  # pragma: no cover
    torch = None

from src.ha.prompts import DISEASES  # noqa: E402

TEMPLATE = "Info:\n{provided_test_results}"


class CharTokenizer:
    pad_token_id, eos_token_id, unk_token_id = 128, 129, None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text if ord(c) < 128]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<u>" + messages[0]["content"] + "<a>"


def tiny_llama():
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(vocab_size=130, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512))


@unittest.skipIf(torch is None, "torch/transformers not installed")
class HeadsTest(unittest.TestCase):
    def test_probs_normalised_and_roundtrip(self):
        from src.contrastive_ha.heads import ContrastiveHeads, load_heads, save_heads

        heads = ContrastiveHeads(16, width=32, depth=2, out_dim=8)
        states, actions = torch.randn(5, 16), torch.randn(4, 16)
        p = heads.probs(states, actions)
        self.assertEqual(tuple(p.shape), (5, 4))
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones(5)))
        self.assertTrue(heads.training)  # probs() restores the mode
        with tempfile.TemporaryDirectory() as tmp:
            save_heads(Path(tmp) / "h.pt", heads, actions, {"emb_dir": "x/base"})
            loaded, acts, meta = load_heads(Path(tmp) / "h.pt")
            self.assertTrue(torch.allclose(loaded.probs(states, acts), p, atol=1e-6))
            self.assertEqual(meta["emb_dir"], "x/base")

    def test_learns_separable_states(self):
        from src.contrastive_ha.heads import ContrastiveHeads

        torch.manual_seed(0)
        centres = torch.randn(4, 16) * 3
        y = torch.randint(0, 4, (400,))
        x = centres[y] + torch.randn(400, 16) * 0.5
        heads = ContrastiveHeads(16, width=32, depth=1, out_dim=8, dropout=0.0)
        opt = torch.optim.AdamW(heads.parameters(), lr=3e-3)
        for _ in range(200):
            loss = heads.loss(x, centres, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertGreater((heads.probs(x, centres).argmax(-1) == y).float().mean().item(), 0.95)


@unittest.skipIf(torch is None, "torch/transformers not installed")
class BackboneScorerTest(unittest.TestCase):
    def setUp(self):
        from src.contrastive_ha.backbone import Backbone
        from src.contrastive_ha.heads import ContrastiveHeads
        from src.contrastive_ha.scorer import ContrastiveHA

        self.model = tiny_llama()
        self.backbone = Backbone(self.model, "base")
        self.heads = ContrastiveHeads(32, width=16, depth=1, out_dim=8)
        self.ha = ContrastiveHA(self.backbone, self.heads, torch.randn(4, 32), CharTokenizer())

    def test_embedding_is_last_token_state(self):
        self.model.train()
        ids = [60, 61, 62, 63]
        emb = self.backbone.embed(ids)
        self.assertTrue(self.model.training)  # restored
        self.model.eval()
        with torch.no_grad():
            ref = self.model.model(input_ids=torch.tensor([ids])).last_hidden_state[0, -1]
        self.assertTrue(torch.allclose(emb, ref, atol=1e-5))
        self.assertEqual(tuple(emb.shape), (32,))

    def test_probs_and_recording(self):
        p = self.ha.probs("Info: pain")
        self.assertAlmostEqual(sum(p), 1.0, places=5)
        self.assertEqual(self.ha.prompt_ids("ab")[-len("Hypothesis:"):], [ord(c) for c in "Hypothesis:"])
        self.assertEqual(self.ha.take(), [])  # not recording outside training
        self.ha.begin("Pancreatitis", recording=True)
        self.ha.probs("Info: pain")
        records = self.ha.take()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][1], DISEASES.index("pancreatitis"))
        self.assertEqual(self.ha.take(), [])

    def test_environment_transcript(self):
        from src.contrastive_ha.scorer import ContrastiveHAEnvironment
        from src.environment.environment import Environment, Trajectory

        base = Environment("{patient_history}{first_hypothesis}", 4096, 13, list(DISEASES), ["CT"],
                           generate_hypothesis=True, hypothesis_prompt_template=TEMPLATE,
                           generate_confidence_calibration=True)
        env = ContrastiveHAEnvironment(base, self.ha)
        t = Trajectory("x{first_hypothesis}", {"CT": "inflamed\n"}, "pain")
        t.provided_tests = ["ct"]
        env._generate_hypothesis([t], None, None, None)
        self.assertRegex(t.text, r"The current hypothesis is: \w+ \(confidence: \d+\)")
        self.assertEqual(len(t.ha_probs), 1)


@unittest.skipIf(torch is None, "torch/transformers not installed")
class CallbackTest(unittest.TestCase):
    def test_update_snapshot_save_and_resume(self):
        from src.contrastive_ha.backbone import Backbone
        from src.contrastive_ha.heads import ContrastiveHeads
        from src.contrastive_ha.scorer import HEADS_FILE, OPTIMIZER_FILE, ContrastiveHA, HeadsUpdateCallback

        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            torch.save({"x": torch.randn(50, 32).half(), "y": torch.randint(0, 4, (50,))}, tmp / "train.pt")
            cfg = SimpleNamespace(update=True, learning_rate=1e-2, weight_decay=0.0, replay=8,
                                  replay_path=str(tmp / "train.pt"), updates_per_step=2, heads_path="start.pt")
            ha = ContrastiveHA(Backbone(tiny_llama(), "base"), ContrastiveHeads(32, 16, 1, 8), torch.randn(4, 32),
                               CharTokenizer())
            cb = HeadsUpdateCallback(ha, cfg, tmp / "run", snapshot_every=2, seed=0)
            ha.begin("appendicitis", recording=True)
            ha.probs("Info: a")
            ha.probs("Info: b")
            before = [p.detach().clone() for p in ha.heads.parameters()]
            state = SimpleNamespace(global_step=2, is_world_process_zero=True)
            cb.on_step_end(None, state, None)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, ha.heads.parameters())))
            line = json.loads((tmp / "run" / "contrastive_heads_updates.jsonl").read_text().splitlines()[0])
            self.assertEqual((line["online_rows"], line["rows"]), (2, 10))
            self.assertTrue((tmp / "run" / "adapters" / "step_0002" / HEADS_FILE).is_file())

            ckpt = tmp / "out" / "checkpoint-2"
            ckpt.mkdir(parents=True)
            cb.on_save(SimpleNamespace(output_dir=str(tmp / "out")), state, None)
            self.assertTrue((ckpt / HEADS_FILE).is_file() and (ckpt / OPTIMIZER_FILE).is_file())

            fresh = ContrastiveHA(Backbone(tiny_llama(), "base"), ContrastiveHeads(32, 16, 1, 8), torch.randn(4, 32),
                                  CharTokenizer())
            cb2 = HeadsUpdateCallback(fresh, cfg, tmp / "run2", snapshot_every=0, seed=0, resume_from=str(ckpt))
            cb2.on_train_begin(None, state, None)
            for a, b in zip(ha.heads.parameters(), fresh.heads.parameters()):
                self.assertTrue(torch.equal(a.cpu(), b.cpu()))

            cfg.update = False
            frozen = HeadsUpdateCallback(fresh, cfg, tmp / "run3", snapshot_every=0, seed=0)
            fresh.begin("appendicitis", recording=True)
            fresh.probs("Info: a")
            snapshot = [p.detach().clone() for p in fresh.heads.parameters()]
            frozen.on_step_end(None, state, None)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(snapshot, fresh.heads.parameters())))
            self.assertEqual(fresh.take(), [])  # records are consumed even when frozen


@unittest.skipIf(torch is None, "torch/transformers not installed")
class TrainHeadsScriptTest(unittest.TestCase):
    def test_end_to_end_on_synthetic_embeddings(self):
        from src.contrastive_ha import train_heads
        from src.utils.episode_log import read_rows, write_rows

        torch.manual_seed(0)
        centres = torch.randn(4, 16) * 3

        def blob(n):
            y = torch.randint(0, 4, (n,))
            return {"x": (centres[y] + torch.randn(n, 16) * 0.5).half(), "y": y}

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            emb = tmp / "emb"
            (emb / "test").mkdir(parents=True)
            torch.save(blob(200), emb / "train.pt")
            torch.save(blob(60), emb / "val.pt")
            torch.save({"x": centres}, emb / "actions.pt")
            test = blob(12)
            rows = [{"stage": "test", "step": 0, "patient_id": i, "episode": 0, "label": DISEASES[int(test["y"][i])],
                     "diagnosis": "", "tests": ["CT"], "cost_usd": 100.0} for i in range(12)]
            write_rows(tmp / "src.csv", rows)
            torch.save(test | {"patient_id": list(range(12)), "source": str(tmp / "src.csv"),
                               "prompt_tokens": [5] * 12}, emb / "test" / "A_da2.pt")
            argv = sys.argv
            sys.argv = ["x", "--emb-dir", str(emb), "--out-dir", str(tmp / "heads"), "--scores-dir",
                        str(tmp / "scores"), "--epochs", "15", "--width", "32", "--out-dim", "8", "--dropout", "0"]
            try:
                train_heads.main()
            finally:
                sys.argv = argv
            summary = json.loads((tmp / "heads" / "summary.json").read_text())
            self.assertGreater(summary["val_accuracy"], 0.9)
            scored = read_rows(tmp / "scores" / "A_da2" / "test_episodes.csv")
            self.assertEqual(len(scored), 12)
            self.assertGreater(sum(r["correct"] for r in scored), 9)
            self.assertTrue((tmp / "scores" / "overview.json").is_file())


if __name__ == "__main__":
    unittest.main()
