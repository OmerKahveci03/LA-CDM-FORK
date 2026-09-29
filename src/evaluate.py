import json
import os
from pathlib import Path

import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from transformers import set_seed

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


@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def main(cfg: DictConfig):
    set_seed(cfg.training.seed)

    setup_wandb(cfg, name_suffix=cfg.evaluation.run_name_suffix)
    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)
    cfg.training.output_dir = output_dir

    # Data, environment, rewards
    test_dataset = prepare_dataset(cfg, "test")
    environment = prepare_environment(cfg)
    reward_functions = prepare_reward_functions(cfg)
    conf_cal_reward_func = prepare_conf_cal_reward_func(cfg)

    # Model and trainer config
    model = load_model(cfg, inference_mode=True)
    grpo_config = configure_grpo(cfg, eval_only=True)

    trainer = CDMGRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=None,
        eval_dataset={"test": test_dataset},
        environment=environment,
        reward_funcs=reward_functions,
        conf_cal_reward_func=conf_cal_reward_func,
        loss_type=cfg.training.loss_type,
        loss_schedule=None,
        test_costs_usd=dict(cfg.evaluation.test_costs_usd),
    )

    # Ensure vLLM engine loads the merged LoRA weights before first generation during the first CDMGRPOTrainer._prepare_inputs call
    if trainer.use_vllm:
        trainer._last_loaded_step = -1

    # Run evaluation on test split
    metrics = trainer.evaluate(eval_dataset="test")

    # Persist metrics
    configured_metrics_path = cfg.evaluation.metrics_output_file
    metrics_path = Path(configured_metrics_path) if configured_metrics_path else Path(output_dir) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    wandb.log({"test_metrics": metrics})
    wandb.finish()


if __name__ == "__main__":
    main()
