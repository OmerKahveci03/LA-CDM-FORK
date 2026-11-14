"""Torch parts of the standalone Hypothesis Agent: the SFT loss and four-way disease scoring.

Scoring replaces free generation: the HA's answer is the disease whose tokens get the
highest log-probability after ``...assistant\\n\\nHypothesis:``. That always gives a valid
disease, a full probability distribution over the four diseases (so exact per-disease AUROC
and a probability-based confidence), and the same procedure for old and new adapters.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F


def suffix_nll(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean NLL of the target suffix of a single unpadded row.

    Asks the model for logits of the last ``n + 1`` positions only (``logits_to_keep``);
    full-vocabulary logits for a 7,000-token prompt would take several GB.
    """
    if input_ids.size(0) != 1:
        raise ValueError("suffix_nll expects batch size 1 (no padding)")
    scored = labels[0] != -100
    n = int(scored.sum())
    if n == 0 or not bool(scored[-n:].all()):
        raise ValueError("The scored tokens must be a non-empty suffix of the row")
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=n + 1).logits
    logits = logits[:, -(n + 1):-1].float()
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, -n:].reshape(-1))


@torch.no_grad()
def candidate_logprobs(model, prompt_ids: list[int], candidates: list[list[int]]) -> list[float]:
    """Sum of log-probabilities of each candidate's tokens after ``prompt_ids``.

    One forward pass over the prompt, then a short pass per candidate that reuses the prompt's
    key/value cache (cropped back after each candidate).
    """
    device = model.get_input_embeddings().weight.device
    out = model(input_ids=torch.tensor([prompt_ids], device=device), use_cache=True, logits_to_keep=1)
    first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    cache, length = out.past_key_values, len(prompt_ids)
    scores = []
    for cand in candidates:
        total = first[cand[0]].item()
        if len(cand) > 1:
            past = cache if hasattr(cache, "crop") else copy.deepcopy(cache)
            step = model(input_ids=torch.tensor([cand[:-1]], device=device), past_key_values=past, use_cache=True)
            logp = torch.log_softmax(step.logits[0].float(), dim=-1)
            total += sum(logp[j, cand[j + 1]].item() for j in range(len(cand) - 1))
            if hasattr(cache, "crop"):
                cache.crop(length)
        scores.append(total)
    return scores


def softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]
