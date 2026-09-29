import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import logging
from transformers import Trainer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from accelerate.utils import (
    gather_object,
    gather,
    broadcast_object_list,
    is_peft_model,
    DistributedType,
    set_seed,
)
from accelerate.utils.other import is_compiled_module
from transformers.training_args import OptimizerNames
from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_torch_xpu_available,
    is_torch_mlu_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_mps_available,
    is_datasets_available,
    is_apex_available,
    is_peft_available,
)
from transformers.trainer_utils import seed_worker
from typing import Any, Dict, Union, Optional

# Datasets
from datasets import Dataset, IterableDataset

# Conditional imports based on optional dependencies
if is_apex_available():
    from apex import amp  # type: ignore

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward  # type: ignore

from collections import defaultdict
from contextlib import nullcontext
import warnings
from unittest.mock import patch

# TRL imports
from trl.trainer.grpo_trainer import RewardFunc
from trl.trainer import GRPOTrainer
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.utils import pad, selective_log_softmax
from trl.data_utils import maybe_apply_chat_template, is_conversational, apply_chat_template
from src.utils.metrics import calculate_classification_metrics
from src.trainer.loss_scheduler import LossScheduler
from src.environment.environment import EnvironmentOutput

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

# PEFT conditional imports
if is_peft_available():
    from peft import PeftConfig, get_peft_model

# vLLM conditional imports
from trl.import_utils import is_vllm_available

if is_vllm_available():
    from vllm import LLM, SamplingParams

from trl.trainer.grpo_config import GRPOConfig


class CDMGRPOTrainer(GRPOTrainer):
    """GRPO trainer extended for the LA-CDM two-agent pipeline.

    Inherits from ``GRPOTrainer`` but **copies** its ``__init__`` rather than
    calling ``super().__init__()`` for two reasons:

    1. ``GRPOTrainer.__init__`` validates that ``per_device_train_batch_size *
       num_processes`` is divisible by ``num_generations``. LA-CDM uses
       ``num_generations`` as the dataloader batch size in order to allow batches
       smaller than the GRPO group size, so this check would
       reject valid configurations.
    2. LA-CDM wraps the vLLM initialization with CUDA-device save/restore to
       prevent vLLM from changing the active device, which cannot be injected
       from outside the parent ``__init__``.

    Pinned to ``trl==0.15.2`` — the copy-pasted init and ``training_step``
    rely on TRL internals that could change across versions.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        environment=None, 
        loss_type="grpo", 
        da_loss_weight=1.0, 
        ha_sft_loss_weight=0.0, 
        ha_conf_cal_loss_weight=0.0, 
        conf_cal_reward_func=None,
        loss_schedule=None,
        test_costs_usd: Optional[dict[str, float]] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if args.beta == 0.0:
            # Evaluation does not compute a KL loss, so a reference-model copy
            # only wastes one full model's worth of GPU memory.
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif not is_peft_model(model):
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        if processing_class.pad_token_id is None:
            processing_class.pad_token = processing_class.eos_token
        model.config.pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.use_vllm = args.use_vllm

        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions

        Trainer.__init__(
            self,
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            # Save the current CUDA device before vLLM initialization, which may change it.
            original_device = torch.cuda.current_device()
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    if torch.cuda.device_count() == 1:
                        vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
                    else:
                        vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )
                with world_size_patch, profiling_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=True,
                        max_model_len=self.args.vllm_max_model_len,
                    )
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
            # Restore the CUDA device after vLLM initialization.
            torch.cuda.set_device(original_device)
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=processing_class.pad_token_id,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        self.environment = environment
        self.sampling_params.stop = ["Observation:"]
        # We want the generation to stop at "Observation:" but we do *not* want
        # the stop string itself to appear in the model output; the environment
        # will add it automatically afterwards.
        self.sampling_params.include_stop_str_in_output = False
        self.loss_type = loss_type
        self.da_loss_weight = da_loss_weight
        self.ha_sft_loss_weight = ha_sft_loss_weight
        self.ha_conf_cal_loss_weight = ha_conf_cal_loss_weight
        self.eval_preds = []
        self.eval_labels = []
        self.eval_provided_tests = []
        self.test_costs_usd = test_costs_usd
        self.train_preds = []
        self.train_labels = []
        self.train_ece_triplets = []
        self.eval_ece_triplets = []
        self.conf_cal_reward_func = conf_cal_reward_func
        self.completion_lengths = []
        self.kls = []
        self.sft_ha_losses = []
        
        # Initialize loss scheduler if configuration is provided
        if loss_schedule is not None:
            self.loss_scheduler = LossScheduler(
                phases=loss_schedule["phases"],
                repeat=loss_schedule.get("repeat", True)
            )
        else:
            self.loss_scheduler = None


    def _mask_after_first_pad(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Create an int mask that is 1 before the first pad token and 0 from the first pad onward.

        Args:
            token_ids: Token ID tensor of shape ``(batch, seq_len)``.

        Returns:
            Int tensor of the same shape with 1s for real tokens and 0s from
            the first pad token onward (inclusive).
        """
        device = token_ids.device
        if token_ids.size(1) == 0:
            return torch.ones_like(token_ids, dtype=torch.int)
        is_pad = token_ids == self.processing_class.pad_token_id
        pad_idx = torch.full((is_pad.size(0),), is_pad.size(1), dtype=torch.long, device=device)
        has_pad = is_pad.any(dim=1)
        if has_pad.any():
            pad_idx[has_pad] = is_pad.int().argmax(dim=1)[has_pad]
        sequence_indices = torch.arange(is_pad.size(1), device=device).expand(is_pad.size(0), -1)
        return (sequence_indices < pad_idx.unsqueeze(1)).int()

    def _run_environment(self, inputs, prompts_text, prompt_ids, device):
        """Run the multi-step environment via vLLM and return padded completion tensors.

        On the main process, executes the environment to produce completions for
        all gathered prompts. Results are broadcast to every process and sliced
        so each process holds only its local portion. Also accumulates
        predictions, labels, and ECE triplets for metric computation.

        Returns:
            Tuple of (env_output, completion_ids, completion_mask,
            prompt_completion_ids, reward_function_kwargs).
        """
        if not self.args.use_vllm:
            raise NotImplementedError("Regular generation not implemented yet")

        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        all_prompts_text = gather_object(prompts_text)
        if self.accelerator.is_main_process:
            env_output = self.environment.run(
                all_prompts_text, inputs, self.llm, self.sampling_params, self.processing_class,
            )
            if self.model.training:
                self.train_preds.extend(kw["diagnosis"] for kw in env_output.reward_function_kwargs)
                self.train_labels.extend(kw["condition"] for kw in inputs)
                self.train_ece_triplets.extend(env_output.ece_calc_triplets)
            else:
                self.eval_preds.extend(kw["diagnosis"] for kw in env_output.reward_function_kwargs)
                self.eval_labels.extend(kw["condition"] for kw in inputs)
                self.eval_ece_triplets.extend(env_output.ece_calc_triplets)
                self.eval_provided_tests.extend(kw["provided_tests"] for kw in env_output.reward_function_kwargs)
        else:
            # Placeholders overwritten by broadcast below; kept so the local
            # variables exist on non-main processes.
            env_output = EnvironmentOutput(
                completion_ids=[None] * len(all_prompts_text),
                completion_mask=[None] * len(all_prompts_text),
                reward_function_kwargs=[None] * len(all_prompts_text),
                ha_prompt_ids=[],
                ha_label_ids=[],
            )

        completion_ids = broadcast_object_list(env_output.completion_ids, from_process=0)
        completion_mask = broadcast_object_list(env_output.completion_mask, from_process=0)
        reward_function_kwargs = broadcast_object_list(env_output.reward_function_kwargs, from_process=0)
        process_slice = slice(
            self.accelerator.process_index * len(inputs),
            (self.accelerator.process_index + 1) * len(inputs),
        )
        completion_ids = completion_ids[process_slice]
        completion_mask = completion_mask[process_slice]
        reward_function_kwargs = reward_function_kwargs[process_slice]

        # Pad completions and concatenate with prompts
        completion_ids = [torch.tensor(ids, device=device, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
        completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
        completion_mask = pad(completion_mask, padding_value=False)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        return env_output, completion_ids, completion_mask, prompt_completion_ids, reward_function_kwargs

    def _prepare_ha_sft_tensors(self, env_output, device):
        """Pad hypothesis agent SFT prompt+label tensors.

        Returns:
            Dict with ``ha_prompt_label_ids``, ``ha_label_mask``,
            ``ha_attention_mask`` (all ``None`` when hypothesis generation is
            disabled).
        """
        if not self.environment.generate_hypothesis:
            return {"ha_prompt_label_ids": None, "ha_label_mask": None, "ha_attention_mask": None}

        ha_prompt_ids = [torch.tensor(ids, device=device, dtype=torch.long) for ids in env_output.ha_prompt_ids]
        ha_label_ids = [torch.tensor(ids, device=device, dtype=torch.long) for ids in env_output.ha_label_ids]
        ha_prompt_label_ids = [torch.cat([p, l], dim=0) for p, l in zip(ha_prompt_ids, ha_label_ids)]
        ha_label_mask = [
            torch.cat([torch.zeros_like(p), torch.ones_like(l)], dim=0)
            for p, l in zip(ha_prompt_ids, ha_label_ids)
        ]
        ha_attention_mask = [torch.ones_like(pl) for pl in ha_prompt_label_ids]
        ha_prompt_label_ids = pad(ha_prompt_label_ids, padding_value=self.processing_class.pad_token_id)
        ha_label_mask = pad(ha_label_mask, padding_value=0)
        ha_attention_mask = pad(ha_attention_mask, padding_value=0)

        return {
            "ha_prompt_label_ids": ha_prompt_label_ids,
            "ha_label_mask": ha_label_mask,
            "ha_attention_mask": ha_attention_mask,
        }

    def _prepare_ha_conf_cal_tensors(self, env_output, inputs, device):
        """Pad confidence calibration tensors, compute ref logprobs, rewards, and advantages.

        Returns:
            Dict with ``ha_conf_cal_*`` tensors (all ``None`` when confidence
            calibration is disabled or no data is available).
        """
        _none = {
            "ha_conf_cal_prompt_ids": None,
            "ha_conf_cal_completion_ids": None,
            "ha_conf_cal_completion_mask": None,
            "ha_conf_cal_completion_attention_mask": None,
            "ha_conf_cal_ref_per_token_logps": None,
            "ha_conf_cal_advantages": None,
        }
        if not self.environment.generate_hypothesis:
            return _none
        if not (self.environment.generate_confidence_calibration and len(env_output.ha_conf_cal_prompt_ids) > 0):
            return _none

        ha_conf_cal_prompt_ids = [torch.tensor(ids, device=device, dtype=torch.long) for ids in env_output.ha_conf_cal_prompt_ids]
        ha_conf_cal_prompt_mask = [torch.ones_like(p) for p in ha_conf_cal_prompt_ids]
        ha_conf_cal_prompt_ids = pad(ha_conf_cal_prompt_ids, padding_value=self.processing_class.pad_token_id)
        ha_conf_cal_prompt_mask = pad(ha_conf_cal_prompt_mask, padding_value=False)

        ha_conf_cal_completion_ids = [torch.tensor(ids, device=device, dtype=torch.long) for ids in env_output.ha_conf_cal_completion_ids]
        ha_conf_cal_completion_ids = pad(ha_conf_cal_completion_ids, padding_value=self.processing_class.pad_token_id)

        ha_conf_cal_prompt_completion_ids = torch.cat([ha_conf_cal_prompt_ids, ha_conf_cal_completion_ids], dim=1)
        ha_conf_cal_completion_mask = self._mask_after_first_pad(ha_conf_cal_completion_ids)
        ha_conf_cal_attention_mask = torch.cat([ha_conf_cal_prompt_mask, ha_conf_cal_completion_mask], dim=1)

        logits_to_keep = ha_conf_cal_completion_ids.size(1)
        with torch.inference_mode():
            if self.ref_model is not None:
                ha_conf_cal_ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, ha_conf_cal_prompt_completion_ids, ha_conf_cal_attention_mask, logits_to_keep,
                )
            else:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                adapter_context = (
                    unwrapped_model.disable_adapter()
                    if hasattr(unwrapped_model, "disable_adapter")
                    else nullcontext()
                )
                with adapter_context:
                    ha_conf_cal_ref_per_token_logps = self._get_per_token_logps(
                        self.model, ha_conf_cal_prompt_completion_ids, ha_conf_cal_attention_mask, logits_to_keep,
                    )

        # Compute confidence calibration rewards and advantages
        ha_conf_cal_completions = self.processing_class.batch_decode(ha_conf_cal_completion_ids, skip_special_tokens=True)
        num_generation_ha = len(ha_conf_cal_prompt_ids)
        rewards_ha_conf_cal = self.conf_cal_reward_func(
            completions=ha_conf_cal_completions, hypothesis=env_output.ha_hypothesis, label=inputs[0]["condition"],
        )
        rewards_ha_conf_cal = torch.tensor(rewards_ha_conf_cal, dtype=torch.float32, device=device)
        mean_grouped = rewards_ha_conf_cal.view(-1, num_generation_ha).mean(dim=1)
        std_grouped = rewards_ha_conf_cal.view(-1, num_generation_ha).std(dim=1, correction=0)
        mean_grouped = mean_grouped.repeat_interleave(num_generation_ha, dim=0)
        std_grouped = std_grouped.repeat_interleave(num_generation_ha, dim=0)
        ha_conf_cal_advantages = (rewards_ha_conf_cal - mean_grouped) / (std_grouped + 1e-4)

        self._metrics["rewards/ha_conf_cal"].append(rewards_ha_conf_cal.mean().item())

        return {
            "ha_conf_cal_prompt_ids": ha_conf_cal_prompt_ids,
            "ha_conf_cal_completion_ids": ha_conf_cal_completion_ids,
            "ha_conf_cal_completion_mask": ha_conf_cal_completion_mask,
            "ha_conf_cal_completion_attention_mask": ha_conf_cal_attention_mask,
            "ha_conf_cal_ref_per_token_logps": ha_conf_cal_ref_per_token_logps,
            "ha_conf_cal_advantages": ha_conf_cal_advantages,
        }

    def _prepare_da_tensors(self, prompts, prompts_text, inputs, env_output,
                            prompt_ids, prompt_mask, completion_ids, completion_mask,
                            prompt_completion_ids, reward_function_kwargs, device):
        """Compute DA pad-masking, ref logprobs, rewards, and advantages.

        Returns:
            Dict with ``no_pad_mask``, ``completion_mask``,
            ``ref_per_token_logps``, ``advantages``, plus ``rewards``,
            ``completions_text``, and ``reward_kwargs`` used for logging.
        """
        no_pad_mask = self._mask_after_first_pad(completion_ids)
        completion_mask = no_pad_mask * completion_mask
        attention_mask = torch.cat([prompt_mask, no_pad_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep,
                )
            else:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                adapter_context = (
                    unwrapped_model.disable_adapter()
                    if hasattr(unwrapped_model, "disable_adapter")
                    else nullcontext()
                )
                with adapter_context:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep,
                    )

        # Decode completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Compute rewards per function
        reward_kwargs = {}
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False,
                )
                reward_inputs = Trainer._prepare_inputs(self, reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                reward_kwargs.update(
                    {key: [example[key] for example in reward_function_kwargs] for key in reward_function_kwargs[0]}
                )
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather, weight, and normalize rewards into advantages
        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1, correction=0)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log reward metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        return {
            "no_pad_mask": no_pad_mask,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "rewards": rewards,
            "completions_text": completions_text,
            "reward_kwargs": reward_kwargs,
        }

    def _log_completions(self, prompts_text, completions_text, reward_kwargs, rewards, env_output):
        """Log a sample of completions to wandb as a table."""
        if not (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            return

        import pandas as pd

        def _list_to_str(item_list):
            """Convert a list (possibly containing None) to a comma-separated string."""
            if item_list is None or len(item_list) == 0:
                return ""
            return ", ".join("" if x is None else str(x) for x in item_list)

        num_rows = len(rewards)

        if (
            env_output.ha_hypotheses_lists is not None
            and len(env_output.ha_hypotheses_lists) == num_rows
        ):
            hypotheses_col = [_list_to_str(hyp_list) for hyp_list in env_output.ha_hypotheses_lists]
        else:
            hypotheses_col = [""] * num_rows

        if (
            env_output.ha_confidences_lists is not None
            and len(env_output.ha_confidences_lists) == num_rows
        ):
            confidences_col = [_list_to_str(conf_list) for conf_list in env_output.ha_confidences_lists]
        else:
            confidences_col = [""] * num_rows

        table = {
            "step": [str(self.state.global_step)] * num_rows,
            "prompt": gather_object(prompts_text),
            "completion": gather_object(completions_text),
            "provided_tests": gather_object(reward_kwargs["provided_tests"]),
            "diagnosis": gather_object(reward_kwargs["diagnosis"]),
            "label": gather_object(reward_kwargs["condition"]),
            "reward": rewards.tolist(),
            "hypotheses": hypotheses_col,
            "confidences": confidences_col,
        }
        df = pd.DataFrame(table)

        if wandb.run is not None and self.accelerator.is_main_process:
            table_key = "train_completions" if self.model.training else "eval_completions"
            wandb.log({table_key: wandb.Table(dataframe=df)})

    def _prepare_inputs(self, inputs):
        """Orchestrate environment execution and tensor preparation for all three loss objectives.

        Runs the multi-step clinical environment, then prepares tensors for:
        - Decision Agent (DA) GRPO loss
        - Hypothesis Agent (HA) SFT loss
        - Hypothesis Agent confidence calibration GRPO loss
        """
        device = "cuda:0"
        inputs = self.environment.construct_prompt(inputs)
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # Run environment and broadcast results
        env_output, completion_ids, completion_mask, prompt_completion_ids, reward_function_kwargs = \
            self._run_environment(inputs, prompts_text, prompt_ids, device)

        # Prepare tensors for each loss objective
        ha_sft = self._prepare_ha_sft_tensors(env_output, device)
        ha_conf_cal = self._prepare_ha_conf_cal_tensors(env_output, inputs, device)
        da = self._prepare_da_tensors(
            prompts, prompts_text, inputs, env_output,
            prompt_ids, prompt_mask, completion_ids, completion_mask,
            prompt_completion_ids, reward_function_kwargs, device,
        )

        self._log_completions(
            prompts_text, da["completions_text"], da["reward_kwargs"], da["rewards"], env_output,
        )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_attention_mask": da["no_pad_mask"],
            "completion_mask": da["completion_mask"],
            "ref_per_token_logps": da["ref_per_token_logps"],
            "advantages": da["advantages"],
            **ha_sft,
            **ha_conf_cal,
        }

    def _compute_loss_da(self, model, inputs, last_minibatch):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask, completion_attention_mask = inputs["completion_ids"], inputs["completion_mask"], inputs["completion_attention_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_attention_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # Note: in newer versions of TRL, the loss is not meaned here.
        if self.loss_type == "grpo":
            token_counts = completion_mask.sum(dim=1).clamp(min=1)  # avoid div-by-zero for empty completions
            loss = ((per_token_loss * completion_mask).sum(dim=1) / token_counts).mean()
        elif self.loss_type == "dr_grpo":
            loss = ((per_token_loss * completion_mask).sum(dim=1) / (per_token_loss.size(0) * self.environment.max_length)).mean()
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self.completion_lengths.append(completion_length)
        if last_minibatch:
            self._metrics["completion_length"].append(torch.mean(torch.tensor(self.completion_lengths)).item())
            self.completion_lengths.clear()

        token_counts_kl = completion_mask.sum(dim=1).clamp(min=1)  # avoid div-by-zero for empty completions
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / token_counts_kl).mean()
        self.kls.append(mean_kl)
        if last_minibatch:
            self._metrics["kl"].append(torch.mean(torch.tensor(self.kls)).item())
            self.kls.clear()

        return loss
    
    def _compute_loss_ha_conf_cal(self, model, inputs, last_minibatch):
        prompt_ids = inputs["ha_conf_cal_prompt_ids"]
        completion_ids, completion_mask, attention_mask = inputs["ha_conf_cal_completion_ids"], inputs["ha_conf_cal_completion_mask"], inputs["ha_conf_cal_completion_attention_mask"]
        if completion_ids is None:
            return 0.0
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ha_conf_cal_ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["ha_conf_cal_advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # Note: in newer versions of TRL, the loss is not meaned here.
        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        elif self.loss_type == "dr_grpo":
            loss = ((per_token_loss * completion_mask).sum(dim=1) / (per_token_loss.size(0) * self.environment.max_length)).mean()
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        return loss
        
    def _compute_loss_ha(self, model, inputs, last_minibatch, chunk_size=1):
        ha_prompt_label_ids = inputs["ha_prompt_label_ids"]
        ha_label_mask = inputs["ha_label_mask"]
        ha_attention_mask = inputs["ha_attention_mask"]

        total_label_tokens = ha_label_mask.sum()
        accumulated_loss = torch.tensor(0.0, device=ha_prompt_label_ids.device)

        B = ha_prompt_label_ids.size(0)
        for i in range(0, B, chunk_size):
            ha_prompt_label_ids_chunk = ha_prompt_label_ids[i : i + chunk_size]
            ha_label_mask_chunk = ha_label_mask[i : i + chunk_size]
            ha_attention_mask_chunk = ha_attention_mask[i : i + chunk_size]

            labels = ha_prompt_label_ids_chunk.clone()
            # This is removing prompt tokens from the loss calculation
            labels = labels.masked_fill(ha_label_mask_chunk == 0, -100)
            # This is removing the confidence value from the loss calculation, since we don't have ground truth here. "0" was used as a placeholder in label generation.
            labels = labels.masked_fill(labels == self.processing_class.encode("0", add_special_tokens=False)[0], -100)

            outputs = model(
                input_ids=ha_prompt_label_ids_chunk,
                attention_mask=ha_attention_mask_chunk,
                labels=labels,
                use_cache=False,
            )
            chunk_loss = outputs.loss
            chunk_tokens = ha_label_mask_chunk.sum()

            accumulated_loss = accumulated_loss + chunk_loss * (chunk_tokens / total_label_tokens)

        self.sft_ha_losses.append(accumulated_loss.item())
        if last_minibatch:
            self._metrics["loss/sft_ha"].append(torch.mean(torch.tensor(self.sft_ha_losses)).item())
            self.sft_ha_losses.clear()

        return accumulated_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, last_minibatch=True):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # Update loss weights using scheduler if available
        if self.loss_scheduler is not None and last_minibatch:
            weights, learning_rate = self.loss_scheduler.step()
            self.da_loss_weight = weights["da_loss_weight"]
            self.ha_sft_loss_weight = weights["ha_sft_loss_weight"]
            self.ha_conf_cal_loss_weight = weights["ha_conf_cal_loss_weight"]
            # If a learning rate is provided by the scheduler, propagate it to the optimizer
            if learning_rate is not None and hasattr(self, "optimizer") and self.optimizer is not None:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = learning_rate
    
        loss_da = self._compute_loss_da(model, inputs, last_minibatch)
        if self.environment.generate_hypothesis:
            loss_ha = self._compute_loss_ha(model, inputs, last_minibatch)
        else:
            self.ha_sft_loss_weight = 0.0
            loss_ha = 0.0

        if self.environment.generate_confidence_calibration:
            loss_ha_conf_cal = self._compute_loss_ha_conf_cal(model, inputs, last_minibatch)
        else:
            self.ha_conf_cal_loss_weight = 0.0
            loss_ha_conf_cal = 0.0

        loss = self.da_loss_weight * loss_da + self.ha_sft_loss_weight * loss_ha + self.ha_conf_cal_loss_weight * loss_ha_conf_cal
        return loss
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """Override that splits the GRPO group into mini-batches for gradient accumulation.

        ``GRPOTrainer`` generates ``num_generations`` completions per prompt,
        but the full group may not fit in GPU memory for the forward/backward
        pass.  This override slices the prepared tensors into chunks of
        ``per_device_train_batch_size`` and accumulates gradients across them,
        then returns the mean loss.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        
        # CHANGED: Split the inputs into mini-batches
        mini_batch_size = self.args.per_device_train_batch_size * self.args.n_gpu
        mini_batch_inputs = []
        for i in range(inputs["prompt_ids"].shape[0] // mini_batch_size):
            slice_dict = {}
            for key, value in inputs.items():
                if value is None:
                    # Keep `None` as-is so downstream code can reliably test for it.
                    slice_dict[key] = None
                elif isinstance(value, torch.Tensor):
                    slice_dict[key] = value[i * mini_batch_size : (i + 1) * mini_batch_size]
            mini_batch_inputs.append(slice_dict)
        losses = []

        del inputs

        # CHANGED: Iterate over the mini-batches for loss calculation and gradient backward pass
        for i, inputs in enumerate(mini_batch_inputs):
            last_minibatch = (i == len(mini_batch_inputs) - 1)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch, last_minibatch=last_minibatch)

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                if is_torch_xpu_available():
                    torch.xpu.empty_cache()
                elif is_torch_mlu_available():
                    torch.mlu.empty_cache()
                elif is_torch_musa_available():
                    torch.musa.empty_cache()
                elif is_torch_npu_available():
                    torch.npu.empty_cache()
                elif is_torch_mps_available(min_version="2.0"):
                    torch.mps.empty_cache()
                else:
                    torch.cuda.empty_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learnign rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # Finally we need to normalize the loss for reporting
                if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                    loss = loss / self.args.gradient_accumulation_steps

                # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
                # https://github.com/huggingface/transformers/pull/35808
                if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                    kwargs["scale_wrt_gas"] = False

                self.accelerator.backward(loss, **kwargs)

            # CHANGED: Append the loss to the list so that we can average it later and return the same value as before
            losses.append(loss.detach())

        # CHANGED: Average the losses and return the same value as before
        losses_device = losses[0].device
        loss = torch.mean(torch.tensor(losses)).to(losses_device)

        return loss.detach()
    
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # Empty completions (e.g. model produced only EOS) have no tokens to score.
        if logits_to_keep == 0:
            return torch.zeros(input_ids.size(0), 0, device=input_ids.device)

        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        # Note: We only add use_cache=False here, to avoid errors in evaluation.
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1, use_cache=False).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens
    
    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
                self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                # Remove base_model and base_layer prefixes
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
                }
                # Remove values with adapter prefix (example: "_lora")
                state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                # When module to save, remove its prefix and discard the original module
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else:
                state_dict = unwrapped_model.state_dict()
            if self.accelerator.is_main_process:
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                # Load weights one-by-one, skipping any that don't match the vLLM model
                # structure (e.g. LoRA-specific parameters). Bulk load_weights() fails on
                # the first mismatch, so individual loading is intentional here.
                for name, weight in state_dict.items():
                    try:
                        llm_model.load_weights(weights=[(name, weight)])
                    except Exception:
                        pass
                logging.disable(logging.INFO)  # Disable INFO and below
                self.llm.llm_engine.reset_prefix_cache()  # Silent
                logging.disable(logging.NOTSET)  # Re-enable all logging
            # Unmerge the adapter to restore the model to its original state.
            # This must be done after loading weights to ensure they correspond to the merged state.
            if is_peft_model(unwrapped_model):
                unwrapped_model.unmerge_adapter()
    
    def log(self, logs: dict[str, float], start_time=None):
        # — existing RL‐style averaging —
        metrics = {k: sum(v)/len(v) for k, v in self._metrics.items()}

        # detect eval vs train
        is_eval = next(iter(logs)).startswith("eval_")
        if is_eval:
            metrics = {f"eval_{k}": v for k, v in metrics.items()}

        # Mutate the Trainer-owned metrics dictionary so evaluate() returns the
        # custom metrics as well as sending them to callbacks/reporters.
        logs.update(metrics)

        if is_eval:
            # — every evaluation run —
            cls = calculate_classification_metrics(
                predictions=self.eval_preds,
                labels=self.eval_labels,
                ece_triplets=self.eval_ece_triplets,
            )
            for k, v in cls.items():
                logs[f"eval_{k}"] = v

            if self.eval_provided_tests and self.test_costs_usd:
                n = len(self.eval_provided_tests)
                total_cost = sum(
                    sum(self.test_costs_usd.get(t.lower(), 0) for t in tests)
                    for tests in self.eval_provided_tests
                )
                logs["eval_avg_diagnostic_cost"] = total_cost / n
                logs["eval_avg_num_tests"] = sum(len(t) for t in self.eval_provided_tests) / n

            # clear eval buffers
            self.eval_preds.clear()
            self.eval_labels.clear()
            self.eval_ece_triplets.clear()
            self.eval_provided_tests.clear()

        # Safely check for end-of-epoch (epoch can be None during test runs)
        epoch_val = self.state.epoch
        try:
            epoch_is_int = float(epoch_val).is_integer()
        except (TypeError, ValueError):
            epoch_is_int = False
        if epoch_val not in (None, 0) and epoch_is_int and len(self.train_preds) > 0:
            # — end of epoch training log —
            cls = calculate_classification_metrics(
                predictions=self.train_preds,
                labels=self.train_labels,
                ece_triplets=self.train_ece_triplets,
            )
            for k, v in cls.items():
                logs[k] = v
            # clear train buffers
            self.train_preds.clear()
            self.train_labels.clear()
            self.train_ece_triplets.clear()

        # call parent & clear step‐level metrics
        super().log(logs, start_time)
        self._metrics.clear()

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self.args.num_generations,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.num_generations,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(test_dataset, Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="test")

        dataloader_params = {
            "batch_size": self.args.num_generations,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(test_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(test_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # We use the same batch_size as for eval.
        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))
