import tempfile
import unittest
from pathlib import Path

from src.utils import episode_log


def record(patient, label, diagnosis, tests=(), end_reason="diagnosis", hypotheses=("appendicitis",)):
    return {
        "patient_id": patient,
        "label": label,
        "diagnosis": diagnosis,
        "end_reason": end_reason,
        "num_turns": len(tests) + 1,
        "provided_tests": list(tests),
        "actions": [f"test:{t}" for t in tests] + ([f"diagnosis:{diagnosis}"] if diagnosis else []),
        "hypotheses": list(hypotheses),
        "confidences": [7] * len(hypotheses),
        "transcript_tokens": 100,
        "last_completion": "Thought: done,\nwith \"quotes\"\nAction: Diagnosis",
        "transcript": "full text",
    }


COSTS = {"CT": 1306, "Ultrasound": 1288, "Complete Blood Count": 71}


class EpisodeLogTest(unittest.TestCase):
    def test_rows_include_rewards_cost_and_episode_index(self):
        rows = episode_log.build_episode_rows(
            [record(7, "Appendicitis", "appendicitis", ["ct", "complete blood count"])] * 2,
            stage="train",
            step=3,
            phase="da",
            reward_names=["DiagnosisReward", "FormatReward", "TestCostReward"],
            rewards_per_func=[[1.0, 1.0, -0.135], [1.0, 1.0, -0.135]],
            advantages=[0.0, 0.0],
            test_costs_usd=COSTS,
        )
        self.assertEqual([r["episode"] for r in rows], [0, 1])
        self.assertTrue(rows[0]["correct"])
        self.assertEqual(rows[0]["cost_usd"], 1377.0)
        self.assertAlmostEqual(rows[0]["reward_total"], 1.865)
        self.assertEqual(rows[0]["final_confidence"], 7)

    def test_csv_round_trip_and_resume_truncation(self):
        rows = []
        for step in (1, 2, 3):
            rows += episode_log.build_episode_rows(
                [record(step, "cholecystitis", None, ["ultrasound"], "invalid_format")],
                stage="train", step=step, test_costs_usd=COSTS,
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_episodes.csv"
            episode_log.append_rows(path, rows[:2])
            episode_log.append_rows(path, rows[2:])
            loaded = episode_log.read_rows(path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[0]["tests"], ["ultrasound"])
            self.assertEqual(loaded[0]["diagnosis"], "")
            self.assertFalse(loaded[0]["correct"])
            self.assertEqual(loaded[0]["last_completion"], rows[0]["last_completion"])
            self.assertEqual(loaded[0]["confidences"], [7.0])

            self.assertEqual(episode_log.truncate_after_step(path, 2), 1)
            self.assertEqual([r["step"] for r in episode_log.read_rows(path)], [1, 2])

    def test_per_disease_summary_and_confusion(self):
        records = [
            record(1, "appendicitis", "appendicitis", ["ct"]),
            record(2, "appendicitis", "cholecystitis", ["ultrasound"]),
            record(3, "appendicitis", None, [], "invalid_format"),
            record(4, "cholecystitis", "cholecystitis", ["ultrasound"], hypotheses=("cholecystitis",)),
        ]
        rows = episode_log.build_episode_rows(records, stage="test", step=0, test_costs_usd=COSTS)
        summary = episode_log.summarize(rows, tests=COSTS)
        appendicitis = summary["per_disease"]["appendicitis"]
        self.assertAlmostEqual(appendicitis["accuracy"], 1 / 3)
        self.assertAlmostEqual(appendicitis["undiagnosed_fraction"], 1 / 3)
        self.assertAlmostEqual(appendicitis["accuracy_diagnosed"], 1 / 2)
        self.assertAlmostEqual(appendicitis["test_rates"]["ct"], 1 / 3)
        self.assertEqual(summary["per_disease"]["cholecystitis"]["precision"], 0.5)
        self.assertEqual(summary["confusion"]["appendicitis"]["undiagnosed"], 1)
        self.assertEqual(summary["overall"]["end_reasons"]["invalid_format"], 0.25)
        self.assertEqual(summary["per_disease"]["diverticulitis"]["n"], 0)

        flat = episode_log.flatten_summary(summary, prefix="eval_")
        self.assertEqual(flat["eval_confusion/appendicitis/cholecystitis"], 1.0)
        self.assertAlmostEqual(flat["eval_disease/appendicitis/accuracy"], 1 / 3)
        self.assertTrue(all(isinstance(v, float) for v in flat.values()))

    def test_ece_matches_binned_definition(self):
        rows = [
            {"label": "appendicitis", "hypotheses": ["appendicitis", "cholecystitis"], "confidences": [8, 8]},
            {"label": "appendicitis", "hypotheses": ["appendicitis", "unknown"], "confidences": [10, 3]},
            {"label": "pancreatitis", "hypotheses": ["pancreatitis"], "confidences": [None]},
        ]
        # Bin 8: two calls, one right -> |0.8 - 0.5|; bin 10: one call, right -> 0.
        # "unknown" and missing confidences are excluded.
        self.assertAlmostEqual(episode_log.expected_calibration_error(rows), (2 / 3) * 0.3)
        self.assertIsNone(episode_log.expected_calibration_error([]))

    def test_validation_curve_orders_steps_and_labels_phases(self):
        import json
        from scripts.validation_curve import load_curve, stage_note

        self.assertEqual(stage_note(0), "untrained")
        self.assertEqual(stage_note(100), "cycle 1 after DA")
        self.assertEqual(stage_note(300), "cycle 1 after calibration")
        self.assertEqual(stage_note(600), "cycle 2 after calibration")
        rows = episode_log.build_episode_rows(
            [record(1, "appendicitis", "appendicitis", ["ct"])], stage="val", step=0, test_costs_usd=COSTS
        )
        with tempfile.TemporaryDirectory() as directory:
            episodes = Path(directory) / "episodes"
            episodes.mkdir()
            for step in (200, 0, 100):
                summary = {"step": step, **episode_log.summarize(rows, tests=COSTS)}
                (episodes / f"val_step_{step:04d}_summary.json").write_text(json.dumps(summary))
            curve = load_curve(Path(directory))
        self.assertEqual([p["step"] for p in curve], [0, 100, 200])
        self.assertEqual(curve[0]["accuracy_appendicitis"], 1.0)
        self.assertEqual(curve[0]["cost_usd"], 1306.0)

    def test_signal_flags_detect_cost_only_advantages(self):
        names = ["DiagnosisReward", "FormatReward", "TestCostReward"]

        def rows_for(components):
            return episode_log.build_episode_rows(
                [record(1, "pancreatitis", None, [], "invalid_format")] * len(components),
                stage="train", step=1, reward_names=names, rewards_per_func=components,
            )

        cost_only = episode_log.group_signal_flags(rows_for([[0, 0, -0.01], [0, 0, 0.0]]))
        self.assertEqual(cost_only["cost_only_signal"], 1.0)
        self.assertEqual(cost_only["no_signal"], 0.0)
        tied = episode_log.group_signal_flags(rows_for([[0, 0, 0.0], [0, 0, 0.0]]))
        self.assertEqual(tied["no_signal"], 1.0)
        self.assertEqual(tied["cost_only_signal"], 0.0)
        mixed = episode_log.group_signal_flags(rows_for([[1, 1, -0.1], [0, 0, 0.0]]))
        self.assertEqual(mixed["cost_only_signal"], 0.0)


if __name__ == "__main__":
    unittest.main()
