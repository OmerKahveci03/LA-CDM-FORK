"""Projection heads of the contrastive HA (pure torch; tests/test_contrastive_ha.py).

As in Contrastive-LM/CLM: a state head and an action head, each an MLP on top of the frozen
backbone's pooled embedding (input LayerNorm, a width-``width`` residual MLP of ``depth``
blocks, a final projection), outputs L2-normalised. The score of action ``a`` for state ``s``
is ``exp(logit_scale) * cos(head_s(s), head_a(a))``; a softmax over the four diseases gives the
HA's probabilities.

Loss: InfoNCE from the state side, i.e. cross-entropy of the true disease against the other
three. The CLM paper also uses the reverse direction (each action against all states in the
batch); with only four actions and many states per disease, that direction has many positives
per action and is left out.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, width: int = 1024, depth: int = 2, out_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.inp = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout))
            for _ in range(depth))
        self.out = nn.Linear(width, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.inp(self.norm(x)))
        for block in self.blocks:
            h = h + block(h)
        return F.normalize(self.out(h), dim=-1)


class ContrastiveHeads(nn.Module):
    def __init__(self, in_dim: int, width: int = 1024, depth: int = 2, out_dim: int = 512, dropout: float = 0.1,
                 init_temperature: float = 0.07):
        super().__init__()
        self.cfg = dict(in_dim=in_dim, width=width, depth=depth, out_dim=out_dim, dropout=dropout)
        self.state_head = ProjectionHead(in_dim, width, depth, out_dim, dropout)
        self.action_head = ProjectionHead(in_dim, width, depth, out_dim, dropout)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / init_temperature)))

    def logits(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``states`` [N, in_dim], ``actions`` [K, in_dim] -> logits [N, K]."""
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        return scale * self.state_head(states) @ self.action_head(actions).T

    def loss(self, states: torch.Tensor, actions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.logits(states, actions), labels)

    @torch.no_grad()
    def probs(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        was = self.training
        self.eval()
        out = torch.softmax(self.logits(states, actions).float(), dim=-1)
        self.train(was)
        return out


def save_heads(path: Path, heads: ContrastiveHeads, actions: torch.Tensor, meta: dict) -> None:
    """Heads + the action embeddings they were trained with (so scoring never re-embeds actions)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": heads.cfg, "state_dict": {k: v.cpu() for k, v in heads.state_dict().items()},
                "actions": actions.detach().cpu(), "meta": meta}, path)


def load_heads(path: Path, device="cpu") -> tuple[ContrastiveHeads, torch.Tensor, dict]:
    blob = torch.load(Path(path), map_location="cpu")
    heads = ContrastiveHeads(**blob["cfg"])
    heads.load_state_dict(blob["state_dict"])
    return heads.to(device), blob["actions"].to(device), blob.get("meta", {})
