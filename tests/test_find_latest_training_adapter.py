import tempfile
import unittest
from pathlib import Path

from scripts.find_latest_training_adapter import find_latest_adapter


class FindLatestAdapterTest(unittest.TestCase):
    def test_ignores_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory)
            complete = outputs / "train_full_complete"
            complete.mkdir()
            (complete / "adapter_config.json").touch()
            (complete / "adapter_model.safetensors").touch()
            incomplete = outputs / "train_full_incomplete"
            incomplete.mkdir()
            (incomplete / "adapter_config.json").touch()
            self.assertEqual(find_latest_adapter(outputs), complete)

    def test_no_completed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(find_latest_adapter(Path(directory)))


if __name__ == "__main__":
    unittest.main()
