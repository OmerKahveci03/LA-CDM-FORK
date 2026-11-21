"""H4 stage 4 (H4c): train a new Decision Agent together with the CLM Hypothesis Agent.

Same as ``src/train_frozen_ha.py`` (unchanged) except for the HA:

- the HA is the CLM HA (``clm.ha_dir``, the stage-3 adapter), loaded as adapter ``clm_ha``
  next to the DA's fresh adapter (src/clm/joint.py);
- the environment is ``CLMHAEnvironment`` (src/clm/environment.py);
- after every DA step the HA is updated on that step's rollouts (``clm.update=true``), or kept
  frozen (``clm.update=false``, the comparison arm);
- the HA adapter is saved with every DA adapter snapshot (``adapters/step_XXXX/clm_ha``).

The loss schedule must contain only ``da`` phases: the shared-adapter HA phases do not apply.
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

from src.clm.environment import CLMHAEnvironment
from src.clm.joint import ADAPTER, CLMHAUpdateCallback, attach_clm_ha
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
    clm = cfg.get("clm")
    if clm is None or not clm.get("ha_dir"):
        raise ValueError("Add '+clm=defaults' and set clm.ha_dir to the CLM HA adapter (stage 3)")
    names = [phase.name for phase in cfg.training.loss_schedule.phases]
    if set(names) != {"da"}:
        raise ValueError(f"Only 'da' phases make sense with a separate CLM HA; got {names}")
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
    score_template = Path(cfg.environment.hypothesis_prompt_template_file).read_text()
    edit_template = Path(clm.edit_template_file).read_text()
    agent = attach_clm_ha(model, tokenizer, str(clm.ha_dir), score_template, edit_template, clm)
    environment = CLMHAEnvironment(prepare_environment(cfg), agent,
                                   notes_log_path=str(output_dir / "clm_notes.jsonl") if clm.notes_log else None)
    da_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ha_weights = sum(p.numel() for p in agent.trainable_parameters())
    print(f"[clm joint] CLM HA {clm.ha_dir} as adapter {ADAPTER!r} ({ha_weights:,} weights, "
          f"{'updated every step' if clm.update else 'frozen'}); DA trainable {da_trainable:,}", flush=True)

    grpo_config = configure_grpo(cfg)
    cycle_steps = sum(phase.steps for phase in cfg.training.loss_schedule.phases)
    guard_cfg = cfg.training.get("collapse_guard") or {}
    collapse_guard = CollapseGuardCallback(
        output_dir,
        min_step=int(guard_cfg.get("min_step", 0)),
        max_undiagnosed=float(guard_cfg.get("max_undiagnosed", 1.0)),
    )
    train_patients = [datasets["train"][i] for i in range(len(datasets["train"]))]
    ha_callback = CLMHAUpdateCallback(agent, environment, train_patients, clm, output_dir, cycle_steps,
                                      cfg.training.seed, resume_from=cfg.training.resume_from_checkpoint)

    trainer = CDMGRPOTrainer(
        model=model,
        args=grpo_config,
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
        # The HA update runs first, so DA snapshots and validation see the updated HA.
        callbacks=[ha_callback, AdapterSnapshotCallback(output_dir, cycle_steps), FinalEvaluationCallback(),
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
            trainer.model.save_pretrained(snapshot)
        (output_dir / COMPLETION_MARKER).write_text(
            json.dumps(
                {
                    "global_step": final_step,
                    "adapter": str(snapshot),
                    "clm_ha_adapter": str(snapshot / ADAPTER),
                    "clm_ha_start": str(clm.ha_dir),
                    "clm_ha_updated": bool(clm.update),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    wandb.finish()


if __name__ == "__main__":
    main()
