"""One update of the CLM Hypothesis Agent (H4 stages 3 and 4).

Two signals, both from rollouts of the HA over a patient's test results:

- **Rewrites, by stepwise GRPO** (the CLM paper, Section 4.2). Rollouts of the same patient
  form a group. Reward 1 if the HA's final hypothesis is correct. The advantage is the GRPO
  outcome advantage plus ``w_eff`` x the success-gated efficiency advantage on the HA's total
  context tokens (only correct rollouts are ranked; see src/clm/context.py). Every HA reply
  (``KEEP`` or a rewrite) in a rollout is trained with that rollout's advantage. One update
  per batch of rollouts, so the policy ratio is 1 and no clipping or reference model is used.
- **Diagnosis, by SFT**: the H2 loss (NLL of " <true disease>" after ``Hypothesis:``) on the
  HA's own context at every turn, plus a replay batch of mixed-disease records, so updates
  from one patient do not all push towards one disease.

The SFT mean is multiplied by ``sft_weight``; ``None`` means 1/B (B = SFT rows from the
rollouts), the deliberate version of attempt 2's weaker HA updates (docs/HYPOTHESIS_H4.md, D6).
"""

from __future__ import annotations

import random

import torch

from src.clm.context import append_only_context, combined_advantages, fit_tests, random_sequence, top_disease
from src.ha.prompts import chat_prompt, encode
from src.ha.scoring import suffix_nll


def completion_logprob(model, prompt_ids: list[int], completion_ids: list[int], device) -> torch.Tensor:
    """Mean log-probability of ``completion_ids`` after ``prompt_ids`` (one unpadded row)."""
    ids = torch.tensor([prompt_ids + completion_ids], device=device)
    n = len(completion_ids)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=n + 1).logits[:, :-1].float()
    targets = ids[:, -n:].to(logits.device)  # the model may be split over two GPUs
    logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return logp.mean()


def replay_rows(agent, patients: list[dict], rng: random.Random, n: int, max_tokens: int = 7000) -> list[tuple]:
    """``n`` mixed records: (scoring prompt ids of an append-only context, label)."""
    if n <= 0 or not patients:
        return []

    def count(message: str) -> int:
        return len(encode(agent.tokenizer, chat_prompt(agent.tokenizer, message)))

    rows = []
    for patient in rng.sample(patients, min(n, len(patients))):
        tests = fit_tests(count, agent.score_template, patient, random_sequence(patient, rng), max_tokens, rng)
        rows.append((agent.score_prompt_ids(append_only_context(patient, tests)), str(patient["condition"]).lower()))
    return rows


def ha_update(agent, groups: list[list[dict]], replay: list[tuple], optimizer, *, w_eff: float = 0.5,
              sft_weight: float | None = None, max_grad_norm: float = 1.0, max_seq_tokens: int = 8192) -> dict:
    """One optimiser step of the HA. ``groups``: lists of rollouts ``{"label", "turns": [TurnRecord]}``."""
    rl, sft = [], []
    successes, costs_all, rewrites, replies = [], [], 0, 0
    for group in groups:
        success = [top_disease(r["turns"][-1].probs, agent.diseases) == r["label"] for r in group]
        costs = [float(sum(t.context_tokens for t in r["turns"])) for r in group]
        advantages = combined_advantages([float(s) for s in success], costs, success, w_eff)
        successes += success
        costs_all += costs
        for advantage, rollout in zip(advantages, group):
            for turn in rollout["turns"]:
                if turn.completion_ids:
                    replies += 1
                    rewrites += turn.action == "rewrite"
                    if advantage != 0.0:
                        rl.append((advantage, turn.edit_prompt_ids, turn.completion_ids))
                sft.append((turn.score_prompt_ids, agent.label_ids(rollout["label"])))
    n_rollout_sft = len(sft)
    sft += [(ids, agent.label_ids(label)) for ids, label in replay]
    weight = sft_weight if sft_weight is not None else 1.0 / max(1, n_rollout_sft)

    params = agent.trainable_parameters()
    rl_loss = sft_nll = 0.0
    skipped = 0
    with agent.training():
        device = agent.device
        optimizer.zero_grad(set_to_none=True)
        for advantage, prompt, completion in rl:
            if len(prompt) + len(completion) > max_seq_tokens:
                skipped += 1
                continue
            loss = -advantage * completion_logprob(agent.model, prompt, completion, device) / len(rl)
            loss.backward()
            rl_loss += loss.item()
        for ids, label in sft:
            seq = torch.tensor([ids + label], device=device)
            labels = torch.tensor([[-100] * len(ids) + label], device=device)
            nll = suffix_nll(agent.model, seq, torch.ones_like(seq), labels)
            (weight * nll / len(sft)).backward()
            sft_nll += nll.item() / len(sft)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    n = max(1, len(successes))
    return {
        "groups": len(groups), "rollouts": len(successes), "success_rate": sum(successes) / n,
        "mean_context_cost": sum(costs_all) / n, "rl_items": len(rl), "rl_skipped_too_long": skipped,
        "rl_loss": rl_loss, "sft_rows": len(sft), "sft_rows_replay": len(replay), "sft_weight": weight,
        "sft_nll": sft_nll, "grad_norm": grad_norm, "rewrite_rate": rewrites / max(1, replies),
    }
