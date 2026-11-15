"""Train only the Decision Agent against a frozen standalone HA (H3, docs/HYPOTHESIS_H3.md).

Same as ``src/train.py`` with three differences:

- the environment's HA turn comes from the frozen HA (``evaluation.frozen_ha_dir``, loaded
  as a second adapter; see src/ha/frozen.py);
- the loss schedule must contain only ``da`` phases (the frozen HA is never trained, and the
  shared adapter no longer plays the HA), which is checked here;
- validation also uses the frozen HA, so checkpoints are chosen on the system being trained.

The DA's LoRA adapter starts fresh, exactly as in ``src/train.py``. Callbacks (adapter
snapshots, final validation, collapse guard) are reused from ``src/train.py``, which is
unchanged.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from transformers import AutoTokenizer, set_seed

from src.environment.frozen_ha_environment import FrozenHAEnvironment
from src.ha.frozen import FrozenHA
from src.train import (
    COMPLETION_MARKER,
    COLLAPSE_MARKER,
    AdapterSnapshotCallback,
    CollapseGuardCallback,
    FinalEvaluationCallback,
    adapter_snapshot_path,
    resumed_step,
)
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


@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def main(cfg: DictConfig):
    frozen_ha_dir = cfg.evaluation.get("frozen_ha_dir")
    if not frozen_ha_dir:
        raise ValueError("Set evaluation.frozen_ha_dir to the frozen HA's adapter folder")
    loss_schedule = cfg.training.loss_schedule
    names = [phase.name for phase in loss_schedule.phases]
    if set(names) != {"da"}:
        raise ValueError(f"With a frozen HA only 'da' phases make sense; got {names}")
    set_seed(cfg.training.seed)

    setup_wandb(cfg)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    cfg.training.output_dir = str(output_dir)
    episode_dir = output_dir / "episodes"

    start_step = resumed_step(cfg.training.resume_from_checkpoint)
    if start_step is not None:
        removed = episode_log.truncate_after_step(episode_dir / "train_episodes.csv", start_step)
        logging.info("Resuming at step %d; removed %d replayed episode rows", start_step, removed)

    datasets = {split: prepare_dataset(cfg, split) for split in ("train", "val")}
    reward_functions = prepare_reward_functions(cfg)
    conf_cal_reward_func = prepare_conf_cal_reward_func(cfg)

    model = load_model(cfg)  # fresh, trainable DA adapter ("default")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    frozen_ha = FrozenHA(model, tokenizer, str(frozen_ha_dir))
    environment = FrozenHAEnvironment(prepare_environment(cfg), frozen_ha.probs, frozen_ha.prompt_ids,
                                      frozen_ha.diseases)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for n, p in model.named_parameters() if f".{frozen_ha.adapter_name}." in n)
    print(f"[frozen HA] {frozen_ha_dir}: {frozen:,} frozen LoRA weights; {trainable:,} trainable (DA)")

    grpo_config = configure_grpo(cfg)
    cycle_steps = sum(phase.steps for phase in loss_schedule.phases)
    guard_cfg = cfg.training.get("collapse_guard") or {}
    collapse_guard = CollapseGuardCallback(
        output_dir,
        min_step=int(guard_cfg.get("min_step", 0)),
        max_undiagnosed=float(guard_cfg.get("max_undiagnosed", 1.0)),
    )

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
        test_costs_usd=dict(cfg.evaluation.test_costs_usd),
        episode_log_dir=str(episode_dir),
        eval_stage_name="val",
        callbacks=[AdapterSnapshotCallback(output_dir, cycle_steps), FinalEvaluationCallback(), collapse_guard],
    )

    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    final_step = trainer.state.global_step
    if collapse_guard.triggered:
        wandb.finish()
        raise SystemExit(f"Training stopped by the collapse guard at step {final_step}; see {COLLAPSE_MARKER}")
    if trainer.is_world_process_zero():
        snapshot = adapter_snapshot_path(output_dir, final_step)
        if not snapshot.exists():
            trainer.model.save_pretrained(snapshot)
        (output_dir / COMPLETION_MARKER).write_text(
            json.dumps(
                {
                    "global_step": final_step,
                    "adapter": str(snapshot),
                    "frozen_ha": str(frozen_ha_dir),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    wandb.finish()


if __name__ == "__main__":
    main()
