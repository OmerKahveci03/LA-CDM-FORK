"""The contrastive HA inside the environment (H4 stages 3 and 4).

``ContrastiveHA.probs(message)`` is a drop-in scorer for the existing
``FrozenHAEnvironment`` (src/environment/frozen_ha_environment.py, unchanged): it embeds the HA
prompt with the frozen backbone, applies the heads and returns the four disease probabilities.
The DA's transcript text is therefore exactly as in step 2 and H3.

``ContrastiveHAEnvironment`` only adds bookkeeping for stage 4: during training (model in train
mode) every embedding is kept with the patient's true disease, and ``HeadsUpdateCallback``
trains the heads on them after each DA step (plus a replay batch of mixed-disease H2 training
records), or leaves them frozen.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from transformers import TrainerCallback

from src.contrastive_ha.backbone import Backbone
from src.contrastive_ha.heads import ContrastiveHeads, load_heads, save_heads
from src.environment.frozen_ha_environment import FrozenHAEnvironment
from src.ha.prompts import ANSWER_PREFIX, DISEASES, chat_prompt, encode

HEADS_FILE = "contrastive_heads.pt"
OPTIMIZER_FILE = "contrastive_heads_optimizer.pt"


class ContrastiveHA:
    def __init__(self, backbone: Backbone, heads: ContrastiveHeads, actions: torch.Tensor, tokenizer,
                 diseases=DISEASES) -> None:
        self.backbone = backbone
        self.device = backbone.device
        self.heads = heads.to(self.device)
        self.actions = actions.to(self.device).float()
        self.tokenizer = tokenizer
        self.diseases = tuple(diseases)
        self.answer_prefix = encode(tokenizer, ANSWER_PREFIX)
        self.recording = False
        self.label: int | None = None
        self.records: list[tuple[torch.Tensor, int]] = []

    @classmethod
    def from_files(cls, model, tokenizer, backbone_kind: str, heads_path: str, h2_adapter_dir: str | None = None):
        backbone = Backbone.attach_h2(model, h2_adapter_dir) if backbone_kind == "h2" else Backbone(model, "base")
        heads, actions, meta = load_heads(heads_path)
        if meta.get("emb_dir") and f"/{backbone_kind}" not in str(meta["emb_dir"]).replace("\\", "/"):
            print(f"[contrastive HA] warning: heads {heads_path} were trained on {meta['emb_dir']}, "
                  f"backbone here is {backbone_kind!r}", flush=True)
        return cls(backbone, heads, actions, tokenizer)

    def prompt_ids(self, message: str) -> list[int]:
        """The H2 HA prompt (as src/ha/frozen.py): chat template + message + ``Hypothesis:``."""
        return encode(self.tokenizer, chat_prompt(self.tokenizer, message)) + self.answer_prefix

    def probs(self, message: str) -> list[float]:
        emb = self.backbone.embed(self.prompt_ids(message))
        if self.recording and self.label is not None:
            self.records.append((emb, self.label))
        return self.heads.probs(emb.to(self.device)[None], self.actions)[0].tolist()

    def begin(self, label: str, recording: bool) -> None:
        self.label = self.diseases.index(label.lower())
        self.recording = recording

    def take(self) -> list[tuple[torch.Tensor, int]]:
        records, self.records = self.records, []
        return records


class ContrastiveHAEnvironment(FrozenHAEnvironment):
    def __init__(self, base, ha: ContrastiveHA) -> None:
        super().__init__(base, ha.probs, ha.prompt_ids, ha.diseases)
        self.ha = ha

    def run(self, prompts_text, inputs, model, sampling_params, processing_class):
        # Trainer.evaluate() puts the model in eval mode: validation/test turns are not recorded.
        self.ha.begin(str(inputs[0]["condition"]), recording=bool(self.ha.backbone.model.training))
        return super().run(prompts_text, inputs, model, sampling_params, processing_class)


class HeadsUpdateCallback(TrainerCallback):
    """Stage 4: update the heads after every DA step (or keep them frozen); save and resume them."""

    def __init__(self, ha: ContrastiveHA, cfg, output_dir: Path, snapshot_every: int, seed: int,
                 resume_from: str | None = None) -> None:
        self.ha = ha
        self.cfg = cfg
        self.update = bool(cfg.update)
        self.output_dir = Path(output_dir)
        self.snapshot_every = snapshot_every
        self.seed = seed
        self.resume_from = Path(resume_from) if resume_from else None
        self.optimizer = torch.optim.AdamW(ha.heads.parameters(), lr=float(cfg.learning_rate),
                                           weight_decay=float(cfg.weight_decay))
        replay = torch.load(cfg.replay_path, map_location="cpu") if cfg.replay_path else None
        self.replay_x = replay["x"].float() if replay else None
        self.replay_y = replay["y"] if replay else None
        self.log_path = self.output_dir / "contrastive_heads_updates.jsonl"

    def _meta(self, step: int) -> dict:
        return {"stage": "joint", "step": step, "updated": self.update, "start_heads": str(self.cfg.heads_path)}

    def on_train_begin(self, args, state, control, **kwargs):
        if self.resume_from is not None:
            path = self.resume_from / HEADS_FILE
            if not path.is_file():
                raise FileNotFoundError(f"Resuming without the heads: {path} is missing")
            heads, _, _ = load_heads(path, device=self.ha.device)
            self.ha.heads.load_state_dict(heads.state_dict())
            if (self.resume_from / OPTIMIZER_FILE).is_file():
                self.optimizer.load_state_dict(torch.load(self.resume_from / OPTIMIZER_FILE, map_location="cpu"))
            print(f"[contrastive HA] resumed the heads from {path}", flush=True)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        records = self.ha.take()
        step = state.global_step
        if self.update and records:
            x = torch.stack([e for e, _ in records])
            y = torch.tensor([label for _, label in records])
            if self.replay_x is not None and int(self.cfg.replay) > 0:
                rng = random.Random(self.seed * 1_000_003 + step)
                idx = rng.sample(range(len(self.replay_y)), min(int(self.cfg.replay), len(self.replay_y)))
                x = torch.cat([x, self.replay_x[idx]])
                y = torch.cat([y, self.replay_y[idx]])
            x, y = x.to(self.ha.device), y.to(self.ha.device)
            heads = self.ha.heads
            heads.train()
            losses = []
            for _ in range(int(self.cfg.updates_per_step)):
                loss = heads.loss(x, self.ha.actions, y)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            heads.eval()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, "online_rows": len(records), "rows": len(y),
                                         "loss_first": losses[0], "loss_last": losses[-1]}) + "\n")
        if self.snapshot_every > 0 and step % self.snapshot_every == 0 and state.is_world_process_zero:
            save_heads(self.output_dir / "adapters" / f"step_{step:04d}" / HEADS_FILE, self.ha.heads,
                       self.ha.actions, self._meta(step))
        return control

    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint.is_dir():
            save_heads(checkpoint / HEADS_FILE, self.ha.heads, self.ha.actions, self._meta(state.global_step))
            torch.save(self.optimizer.state_dict(), checkpoint / OPTIMIZER_FILE)
        return control
