"""Frozen Llama embeddings for the contrastive HA: the last token's final hidden state.

The state is the H2 HA prompt (chat template + ``prompts/hypothesis_agent.txt`` with the
patient's history and test results + ``Hypothesis:``), the same tokens ``src/score_ha.py``
scores. Its last token is where the HA would name the disease, so its hidden state is the
natural pooled representation (CLM also pools the last token).

Two backbones (docs/HYPOTHESIS_H4.md):

- ``base``: Llama-3-8B-Instruct without any adapter. On a PEFT model (e.g. next to the DA's
  adapter) all adapters are disabled during the forward pass.
- ``h2``: the same model with the frozen H2 HA adapter. Standalone, the model *is* the H2
  model; next to the DA it is a second adapter that is switched on for the forward pass only
  (active adapter, LoRA ``requires_grad`` flags and training mode are restored, as in
  src/ha/frozen.py).

The backbone is never trained.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch

BACKBONE_ADAPTER = "contrastive_backbone"


def inner_model(model):
    """The transformer stack (LlamaModel) that returns ``last_hidden_state``; LoRA layers inside
    it follow the PEFT model's active/disabled adapters."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model


class Backbone:
    def __init__(self, model, kind: str, adapter_name: str | None = None):
        if kind not in ("base", "h2"):
            raise ValueError(f"backbone must be 'base' or 'h2', got {kind!r}")
        self.model = model
        self.kind = kind
        self.adapter_name = adapter_name  # set when the H2 adapter sits next to another adapter

    @classmethod
    def attach_h2(cls, model, adapter_dir: str) -> "Backbone":
        """Load the H2 HA as an extra frozen adapter next to the DA's (the DA stays active)."""
        was_training = model.training
        active = model.active_adapter
        model.load_adapter(adapter_dir, adapter_name=BACKBONE_ADAPTER, is_trainable=False)
        model.set_adapter(active)
        model.train(was_training)
        for name, param in model.named_parameters():
            if f".{BACKBONE_ADAPTER}." in name:
                param.requires_grad_(False)
        return cls(model, "h2", BACKBONE_ADAPTER)

    @contextmanager
    def _active(self):
        model = self.model
        was_training = model.training
        is_peft = hasattr(model, "disable_adapter")
        if self.kind == "base" and is_peft:
            switch = model.disable_adapter()
        elif self.kind == "h2" and self.adapter_name is not None:
            switch = self._adapter_switch()
        else:
            switch = nullcontext()
        try:
            model.eval()
            with switch:
                yield
        finally:
            model.train(was_training)

    @contextmanager
    def _adapter_switch(self):
        model = self.model
        active = model.active_adapter
        flags = [(p, p.requires_grad) for n, p in model.named_parameters() if "lora_" in n]
        try:
            model.set_adapter(self.adapter_name)
            yield
        finally:
            model.set_adapter(active)
            for p, flag in flags:
                p.requires_grad_(flag)

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    @torch.no_grad()
    def embed(self, ids: list[int]) -> torch.Tensor:
        """Pooled embedding (float32, CPU) of one unpadded prompt."""
        with self._active():
            out = inner_model(self.model)(input_ids=torch.tensor([ids], device=self.device), use_cache=False)
        return out.last_hidden_state[0, -1].float().cpu()
