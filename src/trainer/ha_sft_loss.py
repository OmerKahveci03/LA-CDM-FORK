"""Hypothesis Agent SFT loss: mean negative log-likelihood of the target tokens.

Kept free of TRL so it can be unit-tested with torch alone.

Each row is ``prompt + "Hypothesis: <true disease>\\nConfidence: 0<eos>"``, right-padded
to the longest row (TRL's ``pad`` default). Only the target suffix is scored, so the
model is asked for logits of just those positions instead of full-vocabulary logits
for the whole prompt. Each row is first trimmed to its real length: scoring "the last
k positions" of a right-padded row would read padding for every row shorter than the
longest, which silently dropped those rows from training (fixed November 2026; the
bug was introduced on September 29 with the suffix-only scoring).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch

# (input_ids [1, L], attention_mask [1, L], logits_to_keep) -> log-probs [1, logits_to_keep]
TokenLogps = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


def trimmed_rows(
    input_ids: torch.Tensor, label_mask: torch.Tensor, attention_mask: torch.Tensor
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield each row with its right padding removed, so its label tokens are the suffix."""
    for i in range(input_ids.size(0)):
        length = int(attention_mask[i].sum().item())
        if length == 0:
            continue
        if not bool(attention_mask[i, :length].all()):
            raise ValueError("HA-SFT rows must be right-padded (attention mask 1s then 0s)")
        yield input_ids[i : i + 1, :length], label_mask[i : i + 1, :length], attention_mask[i : i + 1, :length]


def ha_sft_loss(
    input_ids: torch.Tensor,
    label_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    placeholder_id: int,
    token_logps: TokenLogps,
) -> torch.Tensor:
    """Mean NLL over all scored target tokens of the batch (every row contributes).

    ``placeholder_id`` marks the confidence "0", which is a placeholder and not a target.
    Equals the original paper code (full forward with masked ``labels``, per-chunk token
    means weighted by token counts) up to floating-point rounding.
    """
    scored = label_mask.bool() & (input_ids != placeholder_id)
    total_tokens = scored.sum().clamp(min=1)
    loss = torch.zeros((), device=input_ids.device)
    for ids, row_label_mask, row_attention in trimmed_rows(input_ids, label_mask, attention_mask):
        label_tokens = int(row_label_mask.sum().item())
        if label_tokens == 0:
            continue
        if not bool(row_label_mask[0, -label_tokens:].all()):
            raise ValueError("HA-SFT label tokens must be the suffix of each row")
        row_scored = row_label_mask[:, -label_tokens:].bool() & (ids[:, -label_tokens:] != placeholder_id)
        logps = token_logps(ids, row_attention, label_tokens)
        loss = loss - (logps * row_scored).sum() / total_tokens
    return loss
