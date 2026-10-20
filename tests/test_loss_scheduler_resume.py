import unittest

from src.trainer.loss_scheduler import LossScheduler


class LossSchedulerResumeTest(unittest.TestCase):
    def test_resume_selects_the_next_training_phase(self):
        phases = [
            {"name": "da", "steps": 100, "learning_rate": 1e-5},
            {"name": "ha_sft", "steps": 100, "learning_rate": 1e-5},
            {"name": "ha_conf_cal", "steps": 100, "learning_rate": 1e-5},
        ]
        for completed_steps, expected in [
            (0, "da_loss_weight"),
            (99, "da_loss_weight"),
            (100, "ha_sft_loss_weight"),
            (150, "ha_sft_loss_weight"),
            (200, "ha_conf_cal_loss_weight"),
            (300, "da_loss_weight"),
        ]:
            with self.subTest(completed_steps=completed_steps):
                scheduler = LossScheduler(phases)
                scheduler.seek(completed_steps)
                weights, _ = scheduler.step()
                self.assertEqual(weights[expected], 1.0)
                self.assertEqual(sum(weights.values()), 1.0)
                self.assertEqual(scheduler.current_step, completed_steps + 1)


if __name__ == "__main__":
    unittest.main()
