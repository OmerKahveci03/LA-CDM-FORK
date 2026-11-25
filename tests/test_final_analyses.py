"""A1-A3 helpers (docs/FINAL_ANALYSES.md): threshold choice, Holm, AUROC, ensemble analysis."""

import importlib.util
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DisagreementTest(unittest.TestCase):
    def setUp(self):
        self.a1 = module("scripts/final_disagreement.py", "final_disagreement")

    def test_threshold_choice_prefers_higher_t_on_ties(self):
        # (label, DA diagnosis, HA hypothesis, HA confidence, end reason)
        rows = [("appendicitis", "cholecystitis", "appendicitis", 9.0, "diagnosis"),
                ("appendicitis", "appendicitis", "cholecystitis", 3.0, "diagnosis")]
        t, acc = self.a1.choose_threshold(rows)
        self.assertEqual(acc, 100.0)
        self.assertEqual(t, 9)  # t = 4..9 all give 100%; the highest wins

    def test_holm(self):
        out = self.a1.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(out["a"], 0.03)
        self.assertAlmostEqual(out["c"], 0.06)
        self.assertAlmostEqual(out["b"], 0.06)  # monotone


class EnsembleTest(unittest.TestCase):
    def test_auroc_and_analysis(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from src.contrastive_ha.ensemble_bounds import analyse, auroc

        self.assertEqual(auroc([0.9, 0.8, 0.1], [True, True, False]), 1.0)
        self.assertEqual(auroc([0.5, 0.5], [True, False]), 0.5)
        y = torch.tensor([0, 1, 2, 3])
        confident = torch.tensor([[0.97, 0.01, 0.01, 0.01], [0.01, 0.97, 0.01, 0.01],
                                  [0.01, 0.01, 0.97, 0.01], [0.6, 0.1, 0.1, 0.2]])
        members = torch.stack([confident + 0.001 * k for k in range(5)])
        members = members / members.sum(-1, keepdim=True)
        res = analyse(members, y, resamples=50, seed=0)
        self.assertEqual(res["n"], 4)
        self.assertAlmostEqual(res["ensemble_accuracy"], 0.75)
        self.assertEqual(res["_decisive"], [True, True, True, True])  # tiny spread: every record decisive
        self.assertIsNotNone(res["auroc_wrong"]["mean_confidence"])


if __name__ == "__main__":
    unittest.main()
