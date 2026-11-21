"""H4 stage 4: a new Decision Agent trained while the contrastive HA's heads keep learning.

Same as ``src/train_frozen_ha.py`` (unchanged) except for the HA: the frozen backbone
(``contrastive.backbone``) + the stage-1 heads (``contrastive.heads_path``), updated after every
DA step on that step's HA records plus a mixed replay batch (``contrastive.update=true``), or
kept frozen (``contrastive.update=false``, the comparison arm). The heads are saved next to
every DA snapshot (``adapters/step_XXXX/contrastive_heads.pt``) and with every checkpoint.
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

from src.contrastive_ha.heads import save_heads
from src.contrastive_ha.scorer import HEADS_FILE, ContrastiveHA, ContrastiveHAEnvironment, HeadsUpdateCallback
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


@hydra.main(version_base=None, config_path="../../configs", config_name="defaults")
def main(cfg: DictConfig):
    c = cfg.get("contrastive")
    if c is None or not c.get("heads_path"):
        raise ValueError("Add '+contrastive=defaults' and set contrastive.heads_path")
    names = [phase.name for phase in cfg.training.loss_schedule.phases]
    if set(names) != {"da"}:
        raise ValueError(f"Only 'da' phases make sense with a separate HA; got {names}")
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
    ha = ContrastiveHA.from_files(model, tokenizer, str(c.backbone), str(c.heads_path), str(c.h2_adapter_dir))
    environment = ContrastiveHAEnvironment(prepare_environment(cfg), ha)
    da_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in ha.heads.parameters())
    print(f"[contrastive HA] backbone {c.backbone} (frozen), heads {c.heads_path} ({head_params:,} parameters, "
          f"{'updated every step' if c.update else 'frozen'}); DA trainable {da_trainable:,}", flush=True)

    cycle_steps = sum(phase.steps for phase in cfg.training.loss_schedule.phases)
    guard_cfg = cfg.training.get("collapse_guard") or {}
    collapse_guard = CollapseGuardCallback(
        output_dir,
        min_step=int(guard_cfg.get("min_step", 0)),
        max_undiagnosed=float(guard_cfg.get("max_undiagnosed", 1.0)),
    )
    heads_callback = HeadsUpdateCallback(ha, c, output_dir, cycle_steps, cfg.training.seed,
                                         resume_from=cfg.training.resume_from_checkpoint)

    trainer = CDMGRPOTrainer(
        model=model,
        args=configure_grpo(cfg),
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        environment=environment,
        reward_funcs=reward_functions,
        conf_cal_reward_func=conf_cal_reward_func,
        loss_type=cfg.training.loss_type,
        loss_schedule=cfg.training.loss_schedule,
        test_costs_usd=dict(cfg.evaluation.test_costs_usd),
        episode_log_dir=str(episode_dir),
        eval_stage_name="val",
        # The heads update first, so DA snapshots and validation use the updated heads.
        callbacks=[heads_callback, AdapterSnapshotCallback(output_dir, cycle_steps), FinalEvaluationCallback(),
                   collapse_guard],
    )

    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    final_step = trainer.state.global_step
    if collapse_guard.triggered:
        wandb.finish()
        raise SystemExit(f"Training stopped by the collapse guard at step {final_step}; see {COLLAPSE_MARKER}")
    if trainer.is_world_process_zero():
        snapshot = adapter_snapshot_path(output_dir, final_step)
        if not (snapshot / "adapter_config.json").exists():
            trainer.model.save_pretrained(snapshot, selected_adapters=["default"])
        if not (snapshot / HEADS_FILE).exists():
            save_heads(snapshot / HEADS_FILE, ha.heads, ha.actions, heads_callback._meta(final_step))
        (output_dir / COMPLETION_MARKER).write_text(
            json.dumps(
                {
                    "global_step": final_step,
                    "adapter": str(snapshot),
                    "heads": str(snapshot / HEADS_FILE),
                    "backbone": str(c.backbone),
                    "start_heads": str(c.heads_path),
                    "heads_updated": bool(c.update),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    wandb.finish()


if __name__ == "__main__":
    main()
