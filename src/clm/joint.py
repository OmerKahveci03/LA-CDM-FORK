"""Stage 4 of H4: the CLM HA as a second adapter next to the DA, and its training callback.

The DA keeps the adapter ``default`` (trained by the unchanged CDMGRPOTrainer and merged into
vLLM as before). The CLM HA is the adapter ``clm_ha`` on the same Hugging Face model; it
generates its rewrites with Hugging Face ``generate`` (src/clm/agent.py), so the vLLM weight
sync is untouched (it merges only the active adapter and drops other LoRA weights).

``CLMHAUpdateCallback`` runs after every DA optimiser step:

- takes the step's HA rollouts from the environment and updates the HA (src/clm/learning.py)
  with its own AdamW optimiser (``clm.update=false``: the frozen-CLM-HA arm, no update);
- saves the HA adapter next to every DA adapter snapshot (``adapters/step_XXXX/clm_ha``);
- saves and restores the HA weights and optimiser with every Trainer checkpoint, so a resumed
  job continues exactly.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from transformers import TrainerCallback

from src.clm.agent import CLMHA
from src.clm.learning import ha_update, replay_rows

ADAPTER = "clm_ha"
STATE_FILE = "clm_ha_state.safetensors"
OPTIMIZER_FILE = "clm_ha_optimizer.pt"


def attach_clm_ha(model, tokenizer, adapter_dir: str, score_template: str, edit_template: str, clm_cfg) -> CLMHA:
    """Load the CLM HA as adapter ``clm_ha``; the DA's adapter stays active and trainable."""
    if not hasattr(model, "load_adapter"):
        raise TypeError("The CLM HA needs a PEFT model that already holds the DA's adapter")
    was_training = model.training
    active = model.active_adapter
    model.load_adapter(adapter_dir, adapter_name=ADAPTER, is_trainable=True)
    model.set_adapter(active)
    model.train(was_training)  # load_adapter may switch the model to eval mode
    for name, param in model.named_parameters():
        if f".{ADAPTER}." in name:
            param.requires_grad_(False)  # trained only by CLMHAUpdateCallback, never by the DA's optimiser
    return CLMHA(model, tokenizer, score_template, edit_template, adapter_name=ADAPTER, edit=bool(clm_cfg.edit),
                 max_new_tokens=int(clm_cfg.max_new_tokens), max_words=int(clm_cfg.max_words),
                 temperature=float(clm_cfg.temperature))


def ha_state_dict(model) -> dict:
    from peft import get_peft_model_state_dict

    return {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(model, adapter_name=ADAPTER).items()}


class CLMHAUpdateCallback(TrainerCallback):
    def __init__(self, agent: CLMHA, environment, patients: list[dict], clm_cfg, output_dir: Path,
                 snapshot_every: int, seed: int, resume_from: str | None = None) -> None:
        self.agent = agent
        self.environment = environment
        self.patients = patients
        self.cfg = clm_cfg
        self.output_dir = Path(output_dir)
        self.snapshot_every = snapshot_every
        self.seed = seed
        self.resume_from = Path(resume_from) if resume_from else None
        self.update = bool(clm_cfg.update)
        self.optimizer = torch.optim.AdamW(agent.trainable_parameters(), lr=float(clm_cfg.learning_rate),
                                           weight_decay=0.0)
        self.log_path = self.output_dir / "clm_ha_updates.jsonl"

    def _freeze_ha(self):
        for name, param in self.agent.model.named_parameters():
            if f".{ADAPTER}." in name:
                param.requires_grad_(False)

    def on_train_begin(self, args, state, control, **kwargs):
        if self.resume_from is None:
            return control
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        weights = self.resume_from / STATE_FILE
        if not weights.is_file():
            raise FileNotFoundError(f"Resuming without the CLM HA's weights: {weights} is missing")
        set_peft_model_state_dict(self.agent.model, load_file(str(weights)), adapter_name=ADAPTER)
        if (self.resume_from / OPTIMIZER_FILE).is_file():
            self.optimizer.load_state_dict(torch.load(self.resume_from / OPTIMIZER_FILE, map_location="cpu"))
        self._freeze_ha()
        print(f"[clm joint] resumed the CLM HA from {self.resume_from}", flush=True)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        groups = self.environment.take_groups()
        if self.update and groups:
            step = state.global_step
            rng = random.Random(self.seed * 1_000_003 + step)
            stats = ha_update(self.agent, groups, replay_rows(self.agent, self.patients, rng, int(self.cfg.replay)),
                              self.optimizer, w_eff=float(self.cfg.w_eff),
                              sft_weight=None if self.cfg.sft_weight is None else float(self.cfg.sft_weight),
                              max_seq_tokens=int(self.cfg.max_update_tokens))
            self._freeze_ha()
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, **stats}) + "\n")
        if self.snapshot_every > 0 and state.global_step % self.snapshot_every == 0 and state.is_world_process_zero:
            path = self.output_dir / "adapters" / f"step_{state.global_step:04d}"
            self.agent.model.save_pretrained(str(path), selected_adapters=[ADAPTER])  # -> <path>/clm_ha/
        return control

    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint.is_dir():
            from safetensors.torch import save_file

            save_file(ha_state_dict(self.agent.model), str(checkpoint / STATE_FILE))
            torch.save(self.optimizer.state_dict(), checkpoint / OPTIMIZER_FILE)
        return control
