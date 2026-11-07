import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from transformers import TrainerCallback, set_seed

from src.trainer.cdm_grpo_trainer import CDMGRPOTrainer
from src.utils import episode_log
from src.utils.setup import (
    setup_wandb,
    load_model,
    prepare_dataset,
    prepare_environment,
    prepare_reward_functions,
    prepare_conf_cal_reward_func,
    configure_grpo,
)

ADAPTER_SNAPSHOT_DIR = "adapters"
COMPLETION_MARKER = "training_complete.json"


def adapter_snapshot_path(output_dir: Path, step: int) -> Path:
    return Path(output_dir) / ADAPTER_SNAPSHOT_DIR / f"step_{step:04d}"


class AdapterSnapshotCallback(TrainerCallback):
    """Keep the LoRA adapter at the end of every complete loss-schedule cycle.

    Trainer checkpoints rotate (``save_total_limit``), so cycle-end adapters
    are saved separately to remain available for test evaluation.
    """

    def __init__(self, output_dir: Path, every: int) -> None:
        self.output_dir = Path(output_dir)
        self.every = every

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.every > 0 and state.global_step % self.every == 0 and state.is_world_process_zero:
            path = adapter_snapshot_path(self.output_dir, state.global_step)
            model.save_pretrained(path)
            logging.info("Saved adapter snapshot: %s", path)
        return control


class FinalEvaluationCallback(TrainerCallback):
    """Validate the final model when the last step is not an eval step.

    Trainer only validates at multiples of ``eval_steps``; a run ending at,
    for example, step 1920 would otherwise have no validation of its final
    weights. Trainer saves a checkpoint at the final step by itself.
    """

    def on_step_end(self, args, state, control, **kwargs):
        if (
            state.global_step >= state.max_steps
            and args.eval_strategy == "steps"
            and state.global_step % args.eval_steps != 0
        ):
            control.should_evaluate = True
        return control


def resumed_step(checkpoint: str | None) -> int | None:
    if not checkpoint:
        return None
    state = json.loads((Path(checkpoint) / "trainer_state.json").read_text(encoding="utf-8"))
    return int(state["global_step"])


@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def main(cfg: DictConfig):
    set_seed(cfg.training.seed)

    setup_wandb(cfg)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    cfg.training.output_dir = str(output_dir)
    episode_dir = output_dir / "episodes"

    # A resumed job replays every step after its checkpoint. Remove training
    # episode rows from those steps so each step appears exactly once.
    start_step = resumed_step(cfg.training.resume_from_checkpoint)
    if start_step is not None:
        removed = episode_log.truncate_after_step(episode_dir / "train_episodes.csv", start_step)
        logging.info("Resuming at step %d; removed %d replayed episode rows", start_step, removed)

    # Data, environment, rewards
    datasets = {split: prepare_dataset(cfg, split) for split in ("train", "val")}
    environment = prepare_environment(cfg)
    reward_functions = prepare_reward_functions(cfg)
    conf_cal_reward_func = prepare_conf_cal_reward_func(cfg)

    # Model and trainer config
    model = load_model(cfg)
    grpo_config = configure_grpo(cfg)

    loss_schedule = cfg.training.loss_schedule if hasattr(cfg.training, "loss_schedule") else None
    cycle_steps = sum(phase.steps for phase in loss_schedule.phases) if loss_schedule else 0

    trainer = CDMGRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        environment=environment,
        reward_funcs=reward_functions,
        conf_cal_reward_func=conf_cal_reward_func,
        loss_type=cfg.training.loss_type,
        loss_schedule=loss_schedule,
        # Validation reports real-world test cost alongside accuracy.
        test_costs_usd=dict(cfg.evaluation.test_costs_usd),
        episode_log_dir=str(episode_dir),
        eval_stage_name="val",
        callbacks=[AdapterSnapshotCallback(output_dir, cycle_steps), FinalEvaluationCallback()],
    )

    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    final_step = trainer.state.global_step
    if trainer.is_world_process_zero():
        snapshot = adapter_snapshot_path(output_dir, final_step)
        if not snapshot.exists():
            trainer.model.save_pretrained(snapshot)
        (output_dir / COMPLETION_MARKER).write_text(
            json.dumps(
                {
                    "global_step": final_step,
                    "adapter": str(snapshot),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    wandb.finish()


if __name__ == "__main__":
    main()
