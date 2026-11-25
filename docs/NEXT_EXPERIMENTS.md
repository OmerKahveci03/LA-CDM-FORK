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

## 2. Ablation: DA without an HA ("how does the DA perform on its own?")

**2a (cheap, no training):** run an existing DA (the baseline DA, and H3a's DA) on the test
set with the HA's messages removed from its transcript. That needs one GPU evaluation job per
DA and a "no HA" switch in the environment. It shows how much these DAs *rely* on the HA, but a
DA trained with an HA may simply be confused without one, so it understates a DA trained alone.

**2b (the real ablation):**
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

## 4. The DA reads the HA's full probability distribution (idea from Jev, run locally)

**Origin (November 24, 2026):** TypeSafe AI's **Jev** ("System One" decision model, via Vercel's AI
Gateway; https://typesafe.ai/blog/introducing-system-one-models-and-jev) takes a state plus typed
questions and returns a choice, a probability for every option, and a confidence, without
generating text. The idea was to use Jev as the HA and train a DA on its output.

**Jev itself is not allowed:** it is an external API, and MIMIC records must not be sent to third
parties under the PhysioNet data use agreement. TypeSafe states no data-handling terms, and the
model cannot be self-hosted. Only revisit this with written approval (professor and PhysioNet), or
on non-MIMIC data.

**The local version (no data leaves HiPerGator):** our HAs already compute all four
probabilities and pass on only the top one.
- Change the HA's message from "Hypothesis: appendicitis (confidence 0.62)" to the full
  distribution, for example "appendicitis 0.62, cholecystitis 0.25, pancreatitis 0.10,
  diverticulitis 0.03".
- Train a DA exactly like H3a (frozen separately fine-tuned HA, 700 steps, step chosen on
  validation) and compare it with H3a on the 240 test patients. The CLM HA could be used the same way.
- Question: does seeing the runner-up diagnoses help the DA choose tests, or override the HA less often?
- Needs a new message format (a new environment option, kept separate from the old code), a
  pre-registered decision rule, and about one GPU day.
- Related work: Jev can be cited as an industry example of "state → calibrated choice over typed
  options", but without its (vendor-reported, unverified) performance numbers.

## 5. From the team meeting with Dr. Liu (November 24, 2026)

Source: `ignore/zoom_meeting_notes.md` (automatic meeting summary). **The team aims to submit by
Sunday, November 29.** Xuefeng recommended making the CLM the main research direction for the
paper, rather than the search-based approach.

Ideas raised that are new here:
- **5a. A weaker DA with the CLM HA.** Does a good HA matter more when the DA is weaker? Cheap
  versions need no training: the untrained (zero-shot) DA + CLM HA, or early H4c checkpoints
  (steps 100–300) tested with the CLM HA.
- **5b. Confidence bounds (LCB/UCB) for the DA (Xuefeng).** Give the DA a lower and an upper
  bound for each disease, not one probability. This needs an uncertainty estimate first. The
  cheapest for the CLM HA is an ensemble of heads (retrain the heads with 5–10 seeds on the cached
  embeddings, which takes minutes) and the spread of their probabilities. Then:
  (i) check offline whether the bounds are meaningful (coverage, and whether wide bounds mark
  errors); (ii) a rule such as "diagnose when LCB(top) > UCB(runner-up), otherwise order a test";
  (iii) bounds in the DA's message, which needs DA training.
- **5c. A combined HA.** Merge the separately fine-tuned HA's and the CLM HA's probabilities (or
  feed the baseline HA's output into the CLM) into one hypothesis for the DA. Offline first: average
  the two HAs' probabilities on the H2/H4a record sets (CPU).
- **5d. CLM + agentic search (a three-part design).** The CLM proposes hypotheses; when the top
  options are close, a search agent gathers information to separate them; the refined
  distribution goes to the DA. Piyush noted that the DA's test ordering already is the search,
  so this could be framed as uncertainty-driven test selection (connects to 5b and to H5 in
  section 3). Large; after the submission.
- **5e. Replay of previous patients in DA training.** Suggested against "validation falls after a
  third of the training patients". **Caution:** the large drop after step 700 was the resume
  bug, not overfitting (see `HYPOTHESIS_H4.md`, H4c-ext); tell the team. The smaller swings
  (5–13 points between checkpoints, the frozen arm after step 500) look like single-run
  instability. Revisit only if the corrected extension still degrades.

Already in this file: the disagreement analysis (item 1, which Edward's
`edward/da_ha_disagreement.py` computes) and the ablation without the HA (item 2).

## 6. Smaller items

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
