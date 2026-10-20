import json
import tempfile
import unittest
from pathlib import Path

from scripts.find_latest_training_checkpoint import find_latest_checkpoint


class FindLatestCheckpointTest(unittest.TestCase):
    def test_selects_highest_complete_step_across_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory)
            for job, step, valid in [
                ("old", 100, True),
                ("recent", 150, True),
                ("incomplete", 175, False),
                ("future", 325, True),
            ]:
                checkpoint = outputs / f"train_full_{job}" / f"checkpoint-{step}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps({"global_step": step}), encoding="utf-8"
                )
                if valid:
                    (checkpoint / "optimizer.pt").touch()

            selected = find_latest_checkpoint(outputs, max_steps=300)
            self.assertEqual(selected.name, "checkpoint-150")
            self.assertEqual(selected.parent.name, "train_full_recent")

    def test_no_checkpoint_starts_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(find_latest_checkpoint(Path(directory), max_steps=300))


if __name__ == "__main__":
    unittest.main()
