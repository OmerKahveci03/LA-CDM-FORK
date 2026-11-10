"""Known-answer checks for the statistics in scripts/test_stats.py."""

import importlib.util
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "test_stats", Path(__file__).resolve().parents[1] / "scripts" / "test_stats.py")
stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stats)


class StatsTest(unittest.TestCase):
    def test_auroc(self):
        self.assertEqual(stats.auroc([1, 2, 3, 4], [False, False, True, True]), 1.0)
        self.assertEqual(stats.auroc([4, 3, 2, 1], [False, False, True, True]), 0.0)
        self.assertEqual(stats.auroc([5, 5, 5, 5], [False, True, False, True]), 0.5)  # ties count half
        self.assertAlmostEqual(stats.auroc([1, 2, 3], [True, False, True]), 0.5)
        self.assertIsNone(stats.auroc([1, 2], [True, True]))

    def test_mcnemar_exact(self):
        # 10 discordant pairs split 8/2: two-sided binomial p = 2 * P(X <= 2 | n=10) = 0.109375
        a = [True] * 8 + [False] * 2 + [True] * 5
        b = [False] * 8 + [True] * 2 + [True] * 5
        self.assertEqual(stats.mcnemar_exact(a, b), (8, 2, 0.109375))
        self.assertEqual(stats.mcnemar_exact([True], [True])[2], 1.0)

    def test_holm(self):
        self.assertEqual(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_wilson(self):
        low, high = stats.wilson(205, 240)
        self.assertAlmostEqual(low, 0.804, places=3)
        self.assertAlmostEqual(high, 0.893, places=3)


if __name__ == "__main__":
    unittest.main()
