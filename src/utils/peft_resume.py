"""Restore the trainable LoRA adapter when resuming a run that holds more than one adapter.

The bug (found 2026-11-23, docs/HYPOTHESIS_H4.md): when the model carries a second, frozen
adapter (the H3 ``frozen_ha`` or the H4 ``contrastive_backbone``), ``save_pretrained`` writes the
trainable ``default`` adapter at the checkpoint root and the other one in a subfolder.
``transformers.Trainer._load_from_checkpoint`` then sees the subfolder, loads *only* the
subfolder adapters, and never loads the root ``default`` adapter. Training resumes with the
optimizer state and step counter restored but the DA's LoRA weights freshly initialised, so
the DA behaves like an untrained model again (invalid action formats) and relearns from scratch.

``RestoreAdapterCallback`` reloads the root adapter into ``default`` at ``on_train_begin``,
which runs after the Trainer's own checkpoint loading and before the first generation (so the
first vLLM weight sync already sends the restored weights). The weights are copied in place,
so the optimizer's references to the parameters stay valid.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import TrainerCallback

ROOT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def adapter_norm(model, adapter_name: str = "default") -> float:
    """L2 norm over one adapter's LoRA parameters (a fingerprint to log before/after)."""
    total = 0.0
    for name, param in model.named_parameters():
        if "lora_" in name and f".{adapter_name}." in name:
            total += float(param.detach().float().pow(2).sum())
    return total ** 0.5


def load_root_adapter(model, checkpoint: str | Path, adapter_name: str = "default") -> dict:
    """Copy the checkpoint's root adapter weights into ``adapter_name`` in place."""
    from peft import set_peft_model_state_dict

    checkpoint = Path(checkpoint)
    path = next((checkpoint / f for f in ROOT_FILES if (checkpoint / f).is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No {' or '.join(ROOT_FILES)} in {checkpoint}")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    before = adapter_norm(model, adapter_name)
    result = set_peft_model_state_dict(model, state, adapter_name=adapter_name)
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"{len(unexpected)} checkpoint tensors did not match adapter {adapter_name!r}, "
                           f"e.g. {unexpected[:3]}")
    after = adapter_norm(model, adapter_name)
    return {"path": str(path), "tensors": len(state), "norm_before": before, "norm_after": after}


class RestoreAdapterCallback(TrainerCallback):
    def __init__(self, checkpoint: str | None, adapter_name: str = "default"):
        self.checkpoint = checkpoint
        self.adapter_name = adapter_name

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self.checkpoint or model is None:
            return
        info = load_root_adapter(model, self.checkpoint, self.adapter_name)
        message = (f"[resume] restored adapter '{self.adapter_name}' from {info['path']} "
                   f"({info['tensors']} tensors; LoRA norm {info['norm_before']:.4f} -> {info['norm_after']:.4f})")
        print(message, flush=True)
        logging.info(message)
