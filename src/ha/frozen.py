"""A frozen standalone Hypothesis Agent loaded as a second LoRA adapter (step 2 and H3).

The Decision Agent's adapter stays the active one: vLLM gets the DA's merged weights as
before. The frozen HA is loaded next to it on the same Hugging Face model under the name
``frozen_ha``. ``FrozenHA.probs`` switches to it for one four-way scoring call (the same
scoring as ``src/score_ha.py``), then switches back.

Switching adapters in PEFT also changes which LoRA weights require gradients, and the model
may be in training mode, with dropout and gradient checkpointing switched on. So every call
saves and restores the active adapter, the ``requires_grad`` flags of all LoRA weights and
the training mode. The frozen HA's weights never receive gradients.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from src.ha.prompts import ANSWER_PREFIX, DISEASES, candidate_ids, chat_prompt, encode
from src.ha.scoring import candidate_logprobs, softmax

ADAPTER_NAME = "frozen_ha"


class FrozenHA:
    def __init__(self, model, tokenizer, adapter_dir: str, adapter_name: str = ADAPTER_NAME,
                 diseases=DISEASES) -> None:
        if not hasattr(model, "load_adapter") or not hasattr(model, "set_adapter"):
            raise TypeError("The frozen HA needs a PEFT model that already holds the Decision Agent's adapter")
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_name = adapter_name
        self.diseases = tuple(diseases)
        self._da_adapter = model.active_adapter
        was_training = model.training
        model.load_adapter(adapter_dir, adapter_name=adapter_name, is_trainable=False)
        model.set_adapter(self._da_adapter)  # load_adapter does not activate it; make sure
        # PEFT's load_adapter(is_trainable=False) also puts the whole model in eval mode.
        model.train(was_training)
        for name, param in model.named_parameters():
            if f".{adapter_name}." in name:
                param.requires_grad_(False)
        self.candidates = candidate_ids(tokenizer, self.diseases)
        self.answer_prefix = encode(tokenizer, ANSWER_PREFIX)

    def prompt_ids(self, user_message: str) -> list[int]:
        """Exactly the prompt ``src/score_ha.py`` scores: chat template + ``Hypothesis:``."""
        return encode(self.tokenizer, chat_prompt(self.tokenizer, user_message)) + self.answer_prefix

    @contextmanager
    def _active(self):
        model = self.model
        active = model.active_adapter
        flags = [(p, p.requires_grad) for n, p in model.named_parameters() if "lora_" in n]
        was_training = model.training
        try:
            model.set_adapter(self.adapter_name)
            model.eval()
            yield
        finally:
            model.set_adapter(active)
            for p, flag in flags:
                p.requires_grad_(flag)
            if was_training:
                model.train()

    @torch.no_grad()
    def probs(self, user_message: str) -> list[float]:
        """Probabilities of the diseases (in ``self.diseases`` order) for one HA prompt."""
        with self._active():
            return softmax(candidate_logprobs(self.model, self.prompt_ids(user_message), self.candidates))
