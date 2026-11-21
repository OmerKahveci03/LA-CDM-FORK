"""H4 stage 4: test evaluation of a DA with the CLM HA (the normal test-run outputs).

Same as ``src/evaluate_frozen_ha.py`` (unchanged) except that the HA is the CLM HA
(``clm.ha_dir``, e.g. ``<run>/train/adapters/step_0700/clm_ha``) and every HA turn is written
to ``<output>/clm_notes.jsonl``. The HA replies greedily.
"""

import json
import os
from pathlib import Path

import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from transformers import AutoTokenizer, set_seed

from src.clm.environment import CLMHAEnvironment
from src.clm.joint import ADAPTER, attach_clm_ha
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
    clm = cfg.get("clm")
    if clm is None or not clm.get("ha_dir"):
        raise ValueError("Add '+clm=defaults' and set clm.ha_dir to the CLM HA adapter")
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
    score_template = Path(cfg.environment.hypothesis_prompt_template_file).read_text()
    edit_template = Path(clm.edit_template_file).read_text()
    agent = attach_clm_ha(model, tokenizer, str(clm.ha_dir), score_template, edit_template, clm)
    environment = CLMHAEnvironment(prepare_environment(cfg), agent,
                                   notes_log_path=str(Path(output_dir) / "clm_notes.jsonl"))
    print(f"[clm eval] DA {cfg.evaluation.adapter_dir}; CLM HA {clm.ha_dir} as {ADAPTER!r} "
          f"(edit={bool(clm.edit)})", flush=True)

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
    metrics_path = Path(cfg.evaluation.metrics_output_file or Path(output_dir) / "metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    wandb.log({"test_metrics": metrics})
    wandb.finish()


if __name__ == "__main__":
    main()
