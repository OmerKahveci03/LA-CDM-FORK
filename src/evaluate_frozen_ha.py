"""Evaluate the trained Decision Agent with a frozen standalone HA (step 2, docs/STEP2_COMBINED.md).

Same as ``src/evaluate.py`` except for the Hypothesis Agent's turn. The DA's adapter
(``evaluation.adapter_dir``) is loaded as usual and drives vLLM. The frozen HA
(``evaluation.frozen_ha_dir``) is loaded as a second adapter on the same model and gives
the hypotheses and confidences (``src/environment/frozen_ha_environment.py``).
``src/evaluate.py`` itself is unchanged.
"""

import json
import os
from pathlib import Path

import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from transformers import AutoTokenizer, set_seed

from src.environment.frozen_ha_environment import FrozenHAEnvironment
from src.ha.frozen import FrozenHA
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
    if not cfg.evaluation.get("frozen_ha_dir"):
        raise ValueError("Set evaluation.frozen_ha_dir to the frozen HA's adapter folder")
    if cfg.evaluation.get("scripted_policy"):
        raise ValueError("scripted_policy and frozen_ha_dir cannot be combined")
    if cfg.evaluation.base_model or not cfg.evaluation.adapter_dir:
        raise ValueError("Step 2 needs the trained DA's adapter (evaluation.adapter_dir)")
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
    frozen_ha = FrozenHA(model, tokenizer, str(cfg.evaluation.frozen_ha_dir))
    environment = FrozenHAEnvironment(prepare_environment(cfg), frozen_ha.probs, frozen_ha.prompt_ids,
                                      frozen_ha.diseases)
    print(f"[frozen HA] {cfg.evaluation.frozen_ha_dir} as adapter {frozen_ha.adapter_name!r}; "
          f"DA adapter {model.active_adapter!r} stays active for vLLM")

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
        episode_log_dir=output_dir,
        eval_stage_name="test",
    )
    if trainer.use_vllm:
        trainer._last_loaded_step = -1  # load the DA's merged weights into vLLM before the first turn

    metrics = trainer.evaluate(eval_dataset="test")

    configured_metrics_path = cfg.evaluation.metrics_output_file
    metrics_path = Path(configured_metrics_path) if configured_metrics_path else Path(output_dir) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    wandb.log({"test_metrics": metrics})
    wandb.finish()


if __name__ == "__main__":
    main()
