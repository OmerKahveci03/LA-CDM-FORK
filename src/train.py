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

    setup_wandb(cfg)
    cfg.training.output_dir = HydraConfig.get().runtime.output_dir

    # Data, environment, rewards
    datasets = {split: prepare_dataset(cfg, split) for split in ("train", "val")}
    environment = prepare_environment(cfg)
    reward_functions = prepare_reward_functions(cfg)
    conf_cal_reward_func = prepare_conf_cal_reward_func(cfg)

    # Model and trainer config
    model = load_model(cfg)
    grpo_config = configure_grpo(cfg)

    trainer = CDMGRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        environment=environment,
        reward_funcs=reward_functions,
        conf_cal_reward_func=conf_cal_reward_func,
        loss_type=cfg.training.loss_type,
        loss_schedule=cfg.training.loss_schedule if hasattr(cfg.training, "loss_schedule") else None,
    )

    trainer.train()
    trainer.save_model(cfg.training.output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
