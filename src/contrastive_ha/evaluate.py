"""H4 stages 3-4: a test run of a DA with the contrastive HA (the normal test-run outputs).

Same as ``src/evaluate_frozen_ha.py`` (unchanged) except that the HA is the contrastive HA:
frozen backbone (``contrastive.backbone``) + heads (``contrastive.heads_path``).

- Stage 3: the baseline DA (attempt 2, step 600) + the stage-1 heads.
- Stage 4: a stage-4 DA snapshot + the heads saved with it
  (``train/adapters/step_XXXX/contrastive_heads.pt``).
"""

import json
import os
from pathlib import Path

import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from transformers import AutoTokenizer, set_seed

from src.contrastive_ha.scorer import ContrastiveHA, ContrastiveHAEnvironment
from src.trainer.cdm_grpo_trainer import CDMGRPOTrainer
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
    if cfg.evaluation.base_model or not cfg.evaluation.adapter_dir:
        raise ValueError("Set evaluation.adapter_dir to the DA's adapter")
    set_seed(cfg.training.seed)

    setup_wandb(cfg, name_suffix=cfg.evaluation.run_name_suffix)
    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)
    cfg.training.output_dir = output_dir

    test_dataset = prepare_dataset(cfg, "test")
    reward_functions = prepare_reward_functions(cfg)
    conf_cal_reward_func = prepare_conf_cal_reward_func(cfg)

    model = load_model(cfg, inference_mode=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    ha = ContrastiveHA.from_files(model, tokenizer, str(c.backbone), str(c.heads_path), str(c.h2_adapter_dir))
    environment = ContrastiveHAEnvironment(prepare_environment(cfg), ha)
    print(f"[contrastive HA] DA {cfg.evaluation.adapter_dir}; backbone {c.backbone}; heads {c.heads_path}",
          flush=True)

    trainer = CDMGRPOTrainer(
        model=model,
        args=configure_grpo(cfg, eval_only=True),
        train_dataset=None,
        eval_dataset={"test": test_dataset},
        environment=environment,
        reward_funcs=reward_functions,
        conf_cal_reward_func=conf_cal_reward_func,
        loss_type=cfg.training.loss_type,
        loss_schedule=None,
        test_costs_usd=dict(cfg.evaluation.test_costs_usd),
        episode_log_dir=output_dir,
        eval_stage_name="test",
    )
    if trainer.use_vllm:
        trainer._last_loaded_step = -1  # load the DA's merged weights into vLLM before the first turn

    metrics = trainer.evaluate(eval_dataset="test")
    metrics_path = Path(cfg.evaluation.metrics_output_file or Path(output_dir) / "metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    wandb.log({"test_metrics": metrics})
    wandb.finish()


if __name__ == "__main__":
    main()
