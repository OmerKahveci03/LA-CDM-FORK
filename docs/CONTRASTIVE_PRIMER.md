# Contrastive models: a primer for this project

> **This is the background for the current H4** (`docs/HYPOTHESIS_H4.md`). The meeting's
> "CLM" is the **Contrastive Language Model** of Contrastive-LM/CLM; section 11 describes it and
> our version. Sections 1–10 were written before the repo was identified; where they differ
> from what was built (e.g. the encoder choice in sections 6–8), section 11 is what counts.

A plain-language guide to contrastive models. Read it from top to bottom. Each idea builds
on the one before.

## 1. Embeddings: text as a point in space

An **encoder** turns a piece of text into a list of numbers, a **vector**, for example 768
numbers long. Think of it as a point in a 768-dimensional space.

The goal is that texts **with similar meaning land close together**:
- "CT: inflamed appendix, periappendiceal fat stranding" lands near "acute appendicitis".
- The same report lands far from "gallbladder wall thickening".

A model that does this is an **embedding model**. It does not write text. It only places
texts in that space.

## 2. Measuring closeness: cosine similarity

Two vectors are close if they **point in the same direction**. Cosine similarity measures
this:
- **+1:** same direction (very similar);
- **0:** unrelated;
- **−1:** opposite.

Toy example in 2-D:

| Vector | Value | Similarity to the record |
|---|---|---|
| Patient record | (0.9, 0.4) | — |
| Appendicitis | (1.0, 0.0) | 0.91 |
| Pancreatitis | (0.0, 1.0) | 0.41 |

The record is closer to appendicitis.

## 3. From similarities to probabilities: softmax

Given the record's similarity to each of the four diseases, a **softmax** turns them into
probabilities that add up to 1:

```
p(disease) = exp(sim / τ) / Σ over the 4 diseases of exp(sim / τ)
```

- **τ (tau)** is the **temperature**.
  - A small τ makes the distribution sharp: the top disease gets nearly 100%.
  - A large τ makes it flat: nearer 25% each.
- This gives exactly what the HA must output: a **hypothesis** (the top disease) and a
  **confidence** (its probability).

## 4. Contrastive training: pull the right pair together, push the wrong ones apart

Training shows the model a record together with its true disease and adjusts the encoder
so that:
- the record's vector moves **toward** its true disease's vector;
- it moves **away** from the other diseases' vectors.

The loss is cross-entropy on the softmax above, usually called **InfoNCE**:

```
loss = −log p(true disease)
```

That is the whole idea. "Contrastive" means the model learns by **contrasting** the right
answer with the wrong ones, not by writing anything.

**Variants you will see in papers:**
- **CLIP:** pairs images with captions.
- **Sentence embeddings (SimCSE and similar):** pairs a sentence with its paraphrase.
- **Supervised contrastive:** pulls records **of the same disease** together, with no
  disease vectors. Diagnosis is then "which training records is this one closest to".

## 5. An honest point: with 4 diseases, this is nearly a classifier

If the four disease vectors are simply learned, then "softmax over similarity to 4 vectors"
is mathematically very close to a **classification head** (a linear layer plus softmax).
So the real choice in H4 is **which encoder reads the record, and how**, not the loss.

The contrastive framing still helps in three ways:
- the disease vectors can start from **text descriptions** of each disease;
- the embeddings can be reused, for example by the DA or for nearest-patient lookups;
- τ gives a single, clean knob for calibration.

## 6. Encoders, and how they differ from what we use now

| | Causal LM (Llama-3, our current HA) | Encoder (BERT-style) |
|---|---|---|
| Reads | Left to right; each token sees only earlier ones | The whole text at once, in both directions |
| Trained to | Predict the next token | Fill in masked words, then usually fine-tuned |
| Output | Text, or the probability of the next tokens | One vector per token, pooled into one vector |
| Size here | 8 billion parameters | About 150–400 million |
| Can write text? | Yes | No |

**Our current HA** (H2) is a causal LM. We **read** its probability of
" pancreatitis" etc. after "Hypothesis:". That already gives a 4-way softmax, but from a
model built to write text.

**A contrastive HA** uses an encoder built to **represent** text. It is smaller and faster,
and its whole capacity goes into the mapping "patient state → disease".

## 7. Pooling: from many token vectors to one

An encoder outputs one vector **per token**. To get one vector for the record:
- **CLS pooling:** use the vector of a special first token;
- **mean pooling:** average all the token vectors.

Both are common. Mean pooling is often more robust for long texts.

## 8. The hard part for us: long records ("embed the notes properly")

| Records | Typical length |
|---|---|
| History only | About 500–1,500 tokens |
| All tests | Up to about 7,000 tokens |

Classic clinical encoders such as ClinicalBERT **stop at 512 tokens**, so they would cut off
most of a record. The options:

1. **A long-context encoder** (8,000-token encoders exist, e.g. ModernBERT). This is the
   simplest.
2. **Embed by section, then combine:** embed the history and each test separately, then
   combine the vectors (average, or a small attention layer). Each section stays short, so
   512-token clinical encoders also work.
3. **Llama-3 as the encoder:** take its hidden states (4,096 numbers per token), pool them
   and train a small LoRA contrastively. It handles 8,000 tokens and already understands our
   prompts, but it is big.

A further difficulty: **lab tables are mostly numbers**. Embedding models represent numbers
less well than words. Section-wise embedding, or turning labs into short text such as
"lipase: high (3× upper limit)", may help.

## 9. Confidence and calibration

The softmax gives a probability for each disease, so the confidence comes free, as with the
current HA.
- If the probabilities turn out over- or underconfident, adjust **τ** on the validation set
  (**temperature scaling**).
- This is one number, fitted after training, and it does not change which disease is on
  top.
- It is the standard, cheap calibration fix.

## 10. Learning together with the DA (the H4 idea)

In LA-CDM, both roles share one LoRA adapter, so updating one can undo the other. In H4 the
HA is a **separate model**:
- the DA's GRPO update changes only the DA;
- the HA's contrastive update changes only the HA.

They still **learn together**:
- the HA trains on the records the DA produces;
- the DA learns from an HA that keeps improving.

One trap, seen before in this project: every training step covers **one patient**, so all
the records in that step share **one disease**. Training the HA only on those would push it
toward that disease on every step, the swinging behaviour suspected in attempt 3. The fix is
to add a **replay batch** of records with mixed diseases to every HA update (see the H4
plan).

## Glossary

| Term | Meaning |
|---|---|
| Encoder | A model that turns text into vectors |
| Embedding | The vector for one text |
| Cosine similarity | How aligned two vectors are, from −1 to +1 |
| Softmax | Turns scores into probabilities that sum to 1 |
| Temperature τ | Sharpens or flattens the softmax |
| InfoNCE | Cross-entropy over "which candidate is the true partner" |
| Pooling | Combining per-token vectors into one |
| Temperature scaling | Fitting τ on validation to calibrate the probabilities |
| Replay batch | Stored past examples mixed into each update to keep it balanced |

## 11. The Contrastive Language Model (Contrastive-LM/CLM) and our version

**Their model** (Kwok et al., 2026; https://github.com/Contrastive-LM/CLM): a fast "System
One" decision model.

| Part | Theirs | Ours (`src/contrastive_ha/`) |
|---|---|---|
| State | The agent's situation as text | The HA prompt: the patient's history + test results so far |
| Actions | Candidate actions as text | The four diseases ("Final diagnosis: <disease>") |
| Backbone (frozen) | Qwen3-8B, served by vLLM in "pooling" mode | Our Llama-3-8B-Instruct, plain (`base`) or with the frozen H2 HA adapter (`h2`), run with Hugging Face |
| Pooling | The last token | The last token, at "Hypothesis:", where the HA would name the disease |
| Heads | 20M-parameter projection heads | About 7M each (we have about 7,700 training rows, not about 90M) |
| Loss | InfoNCE, both directions | InfoNCE from the state side (true disease vs the other three) |
| Decision | Highest dot product; softmax gives probabilities | The same: the hypothesis and the confidence |

**Why the backbone stays frozen:** it is the expensive part. Each record is embedded **once**
(stage 1a). Training the heads on the saved vectors then takes minutes. During joint training
with the DA, only the small heads change.

**What vLLM is:** a fast engine for *running* language models. It serves the DA during our
training. The CLM repo uses it to compute embeddings. We do not need it for that: plain
Hugging Face on the GPU that already holds our model does the job.

**Why not their released heads:** they were trained on Qwen3-8B's vectors, so they only fit
Qwen. Using our own Llama means training our own heads with their recipe.

**The pipeline, end to end:**
```
patient history + tests  ──►  HA prompt  ──►  frozen Llama  ──►  last-token vector (4,096)
                                                                   │  state head
                                                                   ▼
"Final diagnosis: appendicitis" … ──► frozen Llama ──► 4 vectors ──► action head
                                                                   │
                                     cosine × scale  ──►  softmax over 4 diseases  ──►  hypothesis + confidence
```
