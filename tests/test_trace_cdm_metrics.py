import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.trace_cdm.metrics import arm_metrics, mcnemar_exact_p, paired_comparison, verify_identical

REPO = Path(__file__).resolve().parents[1]


def episode(case_id, truth, diagnosis, tests=("ultrasound",), cost=1288, actions=None):
    actions = actions or [("test", "ultrasound", "result"), ("diagnosis", diagnosis, "diagnosed")]
    return {"case_id": case_id, "ground_truth": truth, "diagnosis": diagnosis, "invalid": diagnosis is None,
            "num_tests": len(tests), "provided_tests": list(tests), "cost_usd": cost, "critic_calls": 0,
            "steps": [{"action": a, "action_input": i, "outcome": o} for a, i, o in actions]}


REFERENCE = [episode("a", "appendicitis", "appendicitis"), episode("b", "pancreatitis", "appendicitis"),
             episode("c", "cholecystitis", None)]
ARM = [episode("a", "appendicitis", "appendicitis"), episode("b", "pancreatitis", "pancreatitis"),
       episode("c", "cholecystitis", "cholecystitis", tests=("ultrasound", "ct"), cost=2594)]


class MetricsTest(unittest.TestCase):
    def test_arm_metrics(self):
        m = arm_metrics(REFERENCE)
        self.assertAlmostEqual(m["accuracy"], 1 / 3)
        self.assertAlmostEqual(m["undiagnosed_fraction"], 1 / 3)
        self.assertEqual(m["accuracy_among_diagnosed"], 0.5)

    def test_paired(self):
        p = paired_comparison(REFERENCE, ARM, n_boot=200)
        self.assertEqual(p["paired_cases"], 3)
        self.assertAlmostEqual(p["delta_accuracy"], 2 / 3)
        self.assertEqual((p["reference_correct_arm_wrong"], p["reference_wrong_arm_correct"]), (0, 2))
        self.assertLessEqual(p["delta_accuracy_ci95"][0], p["delta_accuracy"])

    def test_mcnemar(self):
        self.assertEqual(mcnemar_exact_p(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact_p(0, 5), 0.0625)

    def test_verify_identical(self):
        self.assertEqual(verify_identical(REFERENCE, REFERENCE), [])
        self.assertTrue(verify_identical(REFERENCE, ARM))


class CompareScriptTest(unittest.TestCase):
    def _write_run(self, root, name, episodes, metrics):
        run = Path(root) / name
        run.mkdir()
        (run / "metrics.json").write_text(json.dumps(metrics))
        if episodes is not None:
            (run / "trace_episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
        return f"{name}={run}"

    def _metrics(self, episodes):
        m = arm_metrics(episodes)
        return {"eval_all/accuracy": m["accuracy"], "eval_none_fraction": m["undiagnosed_fraction"],
                "eval_avg_num_tests": m["mean_tests"], "eval_avg_diagnostic_cost": m["mean_cost_usd"]}

    def _compare(self, runs, root):
        return subprocess.run([sys.executable, "scripts/trace_compare.py", *sum((["--run", r] for r in runs), []),
                               "--output", str(Path(root) / "out"), "--bootstrap", "100"],
                              cwd=REPO, capture_output=True, text=True)

    def test_compare_passes_when_shadow_matches_baseline(self):
        with tempfile.TemporaryDirectory() as root:
            runs = [self._write_run(root, "baseline", None, self._metrics(REFERENCE)),
                    self._write_run(root, "shadow", REFERENCE, self._metrics(REFERENCE)),
                    self._write_run(root, "deterministic", ARM, self._metrics(ARM))]
            result = self._compare(runs, root)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((Path(root) / "out" / "comparison.json").read_text())
            self.assertAlmostEqual(report["paired_vs_reference"]["deterministic"]["delta_accuracy"], 2 / 3)

    def test_compare_fails_loudly_when_shadow_diverges(self):
        with tempfile.TemporaryDirectory() as root:
            runs = [self._write_run(root, "baseline", None, self._metrics(ARM)),
                    self._write_run(root, "shadow", REFERENCE, self._metrics(REFERENCE))]
            result = self._compare(runs, root)
            self.assertEqual(result.returncode, 1)
            self.assertIn("VERIFICATION FAILED", result.stderr)


if __name__ == "__main__":
    unittest.main()
