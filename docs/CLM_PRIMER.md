# Context Language Models (CLMs): a primer for this project

> **Alternative, not the current H4.** This is the *context-editing* CLM. The meeting's "CLM" turned
> out to be a **Contrastive** Language Model; see `docs/HYPOTHESIS_H4.md` and
> `docs/CONTRASTIVE_RUNBOOK.md`. The plan for this alternative is `docs/HYPOTHESIS_H4_CONTEXT_EDITING.md`.

A plain-language guide to the paper `docs/literature/22968_Context_Language_Models.pdf`
(anonymous submission to ICLR 2027) and to what H4 (`docs/HYPOTHESIS_H4_CONTEXT_EDITING.md`) builds from it.
Read from top to bottom.

## 1. What "context" is

A language model only knows what is in its **context**: the text it is given before it
answers. In our project, the HA's context at each turn is:
- the instructions;
- the patient's history;
- every test result so far.

The model reads all of it and answers "Hypothesis: …".

## 2. How normal models handle context: append only

A normal LM, which the paper calls a *standard LM*, only ever **adds to the end**:

```
c(t+1) = c(t) + new text         (Eq. 1 in the paper)
```

Each turn, the new observation and the model's answer are appended. Nothing is removed or
rewritten. Our HA works this way: every test result is added to the record, which only
grows. With all tests, the record reaches about 7,000 tokens.

**The problems with append-only:**
- **The context fills up.** Every model has a token limit: our DA is held to 2,048 tokens,
  and Llama-3-8B to 8,192.
- **Important facts get buried** under long, mostly irrelevant text, such as lab tables full
  of normal values.
- **Everything is re-read every turn**, which costs compute.

## 3. The usual fix, and why the paper dislikes it

Most agent systems use a fixed rule written by humans, a **harness**, for example "when the
context is 75% full, summarise it". The paper shows how this fails: a summary turned "Van
Sant was born in Kentucky (ruled out)" into "birthplace unknown", and the agent then repeated
work it had already done.

## 4. The CLM idea: the model edits its own context

A **Context Language Model** decides **for itself** how its context changes:

```
c(t+1) = f_model(c(t))            (Eq. 2 in the paper: any edit the model chooses)
```

It can:
- **append**, as before (the default);
- **delete** what it no longer needs;
- **rewrite** parts into short notes;
- **keep trackers**, such as "ruled out: …", "best candidate so far: …", "to verify: …".

**How the paper implements it:**
- The live context is mirrored as a **text file**, and the model can edit that file with
  shell commands, as it would edit any file.
- After each edit, the next turn sees the edited context.
- If the model does not edit, new text is appended as usual.

## 5. Why it can work out of the box

Modern models are already good at editing files, from training on code and tools. So
"here is your context as a file; you may edit it" works without any new training on strong
models.

Weaker models need training. Before RL, their 9-billion-parameter model scored six points
**below** a simple summarising harness. Our Llama-3-8B is closer to that weaker case.

## 6. Training a CLM with RL (Section 4.2): two ideas we can reuse

1. **Stepwise GRPO.**
   - Sample a group of complete trajectories for the same task.
   - Compute each trajectory's GRPO advantage from its final outcome.
   - Give that advantage to **every model call** in the trajectory, including the
     context edits.

   This is what our trainer already does for the DA.

2. **The success-gated efficiency advantage** (Eq. 6). Within each group:
   - only **successful** trajectories are compared on cost (in the paper, compute; for us,
     it could be context length or test cost);
   - cheaper ones get a bonus, more expensive ones a penalty, clipped to [−1, 1];
   - failed trajectories get **no** efficiency term, and if fewer than two trajectories
     succeed, nobody does.

   So being cheap **but wrong** is never rewarded. The total advantage is
   A = A_outcome + w · A_efficiency.

**Why idea 2 matters to us beyond the CLM:**
- Our DA's test cost is applied to every valid episode, including wrong ones. In a group
  where all four rollouts are wrong, the cheapest wrong one wins. That is one plausible
  driver of the drift to ordering no tests.
- H3b also gamed a reward by ordering cheap, uninformative tests.

Success gating blocks both.

## 7. What a CLM HA would look like here

At each HA turn:

```
HA context = its own notes so far  +  the new test result
        │
        ▼
the HA may rewrite its notes   →  e.g. "Key: lipase 3× upper limit; CT: peripancreatic
        │                           fat stranding. Ruled out: appendicitis (normal
        │                           appendix on CT)."
        ▼
the hypothesis is read from the (possibly edited) context  →  "Hypothesis: pancreatitis (7)"
```

- Its context becomes a compact **patient-state note** that it maintains itself, instead of
  the full, growing raw record.
- "A mapping between patient state and the final output" (from the meeting notes) fits this:
  the note *is* the patient state, and the hypothesis is read from it.
- The DA's side does not change. It still sees "The current hypothesis is: X (confidence:
  c)".

## 8. Honest fit check

| The paper's setting | Our setting |
|---|---|
| 32,000-token budgets, hundreds of turns, hours per task | 1–3 tests per case; the HA context reaches 7,000 tokens only with all tests |
| Context pressure is the main problem | Context pressure is moderate; the DA's 2,048-token limit does cut some episodes |
| 9–27 billion parameter models, some frontier APIs | Llama-3-8B, local only |

So the gains may be smaller for us. They are most likely where records are long (many
tests, lab-heavy) and in **what the notes keep**: a clean "ruled out / key findings" state
may help the HA reason even when the context is short. H4 measures this before any expensive
training.

**A data rule:** MIMIC records must not be sent to external LLM services under PhysioNet's
credentialed-data rules. Any notes must be written by our own local model, never by an API
model.

## Glossary

| Term | Meaning |
|---|---|
| Context | Everything the model reads before answering |
| Append-only | New text is only ever added to the end |
| Harness | Fixed, human-written code that manages the context (e.g. summarise at 75%) |
| CLM | A model that edits its own context: delete, rewrite, keep notes |
| Context as a file | The paper's implementation: the context is mirrored to a file the model can edit |
| Compaction | Rewriting a long context into a shorter one |
| Stepwise GRPO | GRPO where every model call in a trajectory gets that trajectory's advantage |
| Success-gated efficiency | A cost bonus or penalty that applies only among successful trajectories |
