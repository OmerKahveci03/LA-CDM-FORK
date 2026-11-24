# Next experiments (ideas backlog)

**Written November 23, 2026.** Nothing here has been run or pre-registered yet. Each idea needs
its own written plan and decision rule before any result. The H4 extension status is in
`docs/H4_EXTENSION_RESTART.md`.

## 1. DA–HA disagreement analysis (planned for Wednesday, November 25)

**Why:** in every system, the HA's final hypothesis (87–89% on test) is as accurate as or more
accurate than the DA's own diagnosis (82–86%). The DA sometimes overrides the HA.

**Step 1: CPU only, on existing episodes.** For each system, tabulate:

|                | HA right                      | HA wrong                     |
|----------------|-------------------------------|------------------------------|
| DA agrees      | both right                    | both wrong                   |
| DA disagrees   | DA overrode a correct HA (bad) | DA corrected a wrong HA (good) |

Also plot DA accuracy against an "defer to the HA when its confidence ≥ t" rule, for every t.

- Systems: baseline model (`outputs/cost_sweep/cost_1_attempt2/test_step_0600`),
  baseline DA + separately fine-tuned HA (`.../frozen_ha_seed0_step_0600`), H3a
  (`outputs/h3/h3a_ha0/frozen_ha_seed0_step_0700`), H4b (`outputs/contrastive/combined/h2`),
  H4c joint/frozen (`outputs/contrastive/joint/{joint,frozen}_h2/test_step_{0600,0400}`).
- **Choose any rule (for example the threshold t) on validation episodes**
  (`train/episodes/val_step_XXXX_episodes.csv` at each run's chosen step), then evaluate it
  once on test. Paired McNemar against the DA's own diagnosis.

**Then decide:**
- "Corrected a wrong HA" is rare → the DA's overrides are noise; the result is the rule
  "diagnosis = the HA's final hypothesis" (no training needed).
- Both disagreement cells are sizeable and the HA's confidence separates them → a
  confidence-gated deferral rule.
- Only if no simple rule captures it → consider a reward experiment (GPU).

**Warning about rewards:** do not reward *agreement*. The DA would learn to copy the HA, which
the free rule already does, and H3b showed that rewards built on the HA's output get gamed. A
reward for *correctly overriding* is mostly already in the diagnosis reward; any extra term
must be capped so it cannot dominate the diagnosis reward.

## 2. Ablation: DA without an HA

We have never tested whether the HA helps the DA. Train a DA exactly like H3a (same reward,
700 steps, validation-chosen step) but with no hypothesis in its transcript, and compare it
with H3a on the 240 test patients. This needs a "no HA" switch in the environment and about
one GPU day. It tells whether the HA's value is in advising the DA or in making the diagnosis.

## 3. H5: a CLM-style DA (contrastive heads choose the next test)

- State = the DA's record so far, embedded by frozen Llama; actions = 12 tests and "stop";
  the policy = softmax over the heads' scores, with ordered tests masked out.
- On "stop", the diagnosis is the HA's hypothesis.
- Train only the heads with RL (GRPO/REINFORCE), same reward (diagnosis minus cost).
- Pros: no invalid formats, so none of the format-collapse instability; embeddings can be
  cached, so it trains in hours on one GPU. Closer to "map the patient's state to the decision".
- Risks: no longer a text-reasoning agent; the record vector may not show what is missing;
  sparse rewards.
- About one day to build. Compare with H3a.

## 4. Smaller items

- CLM heads learned *from scratch* jointly with the DA (closest to the paper's joint design;
  H4c started from pre-trained heads).
- Add `RestoreAdapterCallback` to `src/train_frozen_ha.py` (the H3 script still has the
  resume bug).
- Seeds: repeat H3a and the H4c arms with another seed (single runs cannot resolve 2 points).
- Week 9 report: consider adding that the chosen CLM HA is built on the separately
  fine-tuned HA, so its accuracy is largely inherited (a CLM on plain Llama reached 85.4%), and
  that H3b's checkpoint was effectively chosen among steps 100–700.

**Suggested order:** 1 (CPU) → 2 (one GPU run) → 3 (H5). Item 1's answer decides how H5 should
diagnose.
