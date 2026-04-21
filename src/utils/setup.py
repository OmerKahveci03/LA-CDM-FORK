"""Shared setup utilities for training and evaluation entry points."""

import torch
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import GRPOConfig
from accelerate import PartialState

from src.dataset.mimic_cdm_dataset import MIMICCDMDataset
from src.environment.environment import Environment


def setup_wandb(cfg: DictConfig, name_suffix: str = "") -> None:
    """Initialize Weights & Biases logging.

    Args:
        cfg: Hydra config.
        name_suffix: Optional suffix appended to the run name (e.g. "-eval").
    """
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.experiment.name + name_suffix,
        tags=cfg.experiment.tags,
        mode=cfg.wandb.mode,
        save_code=cfg.wandb.save_code,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )


def load_model(cfg: DictConfig, inference_mode: bool = False) -> PreTrainedModel:
    """Load base model with optional LoRA and quantization.

    Args:
        cfg: Hydra config.
        inference_mode: If True and an adapter_dir is configured, loads
            pre-trained adapter weights for evaluation instead of creating
            fresh LoRA parameters.

    Returns:
        The model ready for training or inference.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    device_map = {"": PartialState().process_index}
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        quantization_config=bnb_config if cfg.model.quant_4bit else None,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if cfg.model.quant_4bit:
        model = prepare_model_for_kbit_training(model)

    if inference_mode:
        # Load pre-trained adapter weights for evaluation
        if not cfg.evaluation.base_model and cfg.evaluation.adapter_dir is not None and str(cfg.evaluation.adapter_dir).strip() != "":
            model = PeftModel.from_pretrained(model, cfg.evaluation.adapter_dir)
    else:
        # Training mode: enable gradients and create fresh LoRA
        model.enable_input_require_grads()
        if cfg.training.use_vllm:
            model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            target_modules=list(cfg.model.lora_target_modules),
            lora_dropout=cfg.model.lora_dropout,
            bias=cfg.model.lora_bias,
            task_type=cfg.model.lora_task_type,
            inference_mode=cfg.model.lora_inference_mode,
        )
        model = get_peft_model(model, lora_config)

    return model


def prepare_dataset(cfg: DictConfig, split: str) -> MIMICCDMDataset:
    """Create a MIMICCDMDataset for the given split.

    Args:
        cfg: Hydra config.
        split: One of 'train', 'val', or 'test'.

    Returns:
        The dataset for the requested split.
    """
    data_files = {
        "train": cfg.dataset.train_data_file,
        "val": cfg.dataset.val_data_file,
        "test": cfg.dataset.test_data_file,
    }
    return MIMICCDMDataset(
        data_file=data_files[split],
        lab_test_mapping_file=cfg.dataset.lab_test_mapping_file,
        lab_tests=cfg.dataset.lab_tests,
        imaging_tests=cfg.dataset.imaging_tests,
        other_tests=cfg.dataset.other_tests,
        small_sample=cfg.dataset.small_sample,
    )


def prepare_environment(cfg: DictConfig) -> Environment:
    """Build Environment from config (loads prompt templates).

    Args:
        cfg: Hydra config.

    Returns:
        Configured Environment instance.
    """
    with open(cfg.environment.prompt_template_file, "r") as f:
        prompt_template = f.read()

    hypothesis_prompt_template = None
    if cfg.environment.generate_hypothesis:
        with open(cfg.environment.hypothesis_prompt_template_file, "r") as f:
            hypothesis_prompt_template = f.read()

    return Environment(
        prompt_template=prompt_template,
        max_length=cfg.environment.max_length,
        max_steps=cfg.environment.max_steps,
        disease_list=cfg.dataset.disease_list,
        test_list=cfg.dataset.lab_tests + cfg.dataset.imaging_tests + cfg.dataset.other_tests,
        generate_hypothesis=cfg.environment.generate_hypothesis,
        hypothesis_prompt_template=hypothesis_prompt_template,
        generate_confidence_calibration=cfg.environment.generate_confidence_calibration,
        seed=cfg.training.seed,
    )


def prepare_reward_functions(cfg: DictConfig) -> list:
    """Instantiate reward functions from Hydra config.

    Args:
        cfg: Hydra config.

    Returns:
        List of reward function instances.
    """
    return [hydra.utils.instantiate(rf) for rf in cfg.reward_function.reward_functions]


def prepare_conf_cal_reward_func(cfg: DictConfig):
    """Instantiate the confidence calibration reward function.

    Args:
        cfg: Hydra config.

    Returns:
        Confidence calibration reward function instance.
    """
    return hydra.utils.instantiate(cfg.reward_function.conf_cal_reward_func)


def configure_grpo(cfg: DictConfig, eval_only: bool = False) -> GRPOConfig:
    """Build GRPOConfig from Hydra config.

    Args:
        cfg: Hydra config.
        eval_only: If True, disables training-specific settings (logging,
            saving, evaluation strategy) and sets num_generations=1.

    Returns:
        Configured GRPOConfig instance.
    """
    if eval_only:
        grpo_config = GRPOConfig(
            run_name=cfg.experiment.name + cfg.evaluation.run_name_suffix,
            output_dir=cfg.training.output_dir,
            report_to=cfg.training.report_to,
            logging_strategy="no",
            save_strategy="no",
            eval_strategy="no",
            log_completions=cfg.training.log_completions,
            per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
            num_generations=1,
            max_completion_length=cfg.training.max_completion_length,
            max_prompt_length=cfg.training.max_prompt_length,
            temperature=getattr(cfg.training, "temperature", 1.0),
            bf16=cfg.training.bf16,
            bf16_full_eval=cfg.training.bf16_full_eval,
            use_vllm=cfg.training.use_vllm,
            vllm_gpu_memory_utilization=cfg.training.vllm_gpu_memory_utilization,
            seed=cfg.training.seed,
        )
    else:
        grpo_config = GRPOConfig(
            # Output and logging settings
            run_name=cfg.experiment.name,
            output_dir=cfg.training.output_dir,
            report_to=cfg.training.report_to,
            logging_strategy=cfg.training.logging_strategy,
            logging_steps=cfg.training.logging_steps,
            save_strategy=cfg.training.save_strategy,
            save_steps=cfg.training.save_steps,
            save_total_limit=cfg.training.save_total_limit,
            load_best_model_at_end=cfg.training.load_best_model_at_end,
            metric_for_best_model=cfg.training.metric_for_best_model,
            log_completions=cfg.training.log_completions,
            eval_strategy=cfg.training.eval_strategy,
            eval_steps=cfg.training.eval_steps,
            eval_on_start=cfg.training.eval_on_start,
            # Training hyperparameters
            learning_rate=cfg.training.learning_rate,
            num_train_epochs=cfg.training.num_train_epochs,
            lr_scheduler_type=cfg.training.lr_scheduler_type,
            beta=cfg.training.beta,
            # Batch and gradient settings
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
            gradient_checkpointing=cfg.training.gradient_checkpointing,
            # Generation parameters
            max_completion_length=cfg.training.max_completion_length,
            max_prompt_length=cfg.training.max_prompt_length,
            num_generations=cfg.training.num_generations,
            # Hardware optimization
            bf16=cfg.training.bf16,
            bf16_full_eval=cfg.training.bf16_full_eval,
            use_vllm=cfg.training.use_vllm,
            vllm_gpu_memory_utilization=cfg.training.vllm_gpu_memory_utilization,
            seed=cfg.training.seed,
        )

    # Reserve one GPU for vLLM's separate inference engine
    if cfg.training.use_vllm and torch.cuda.device_count() > 1:
        grpo_config._n_gpu -= 1

    return grpo_config