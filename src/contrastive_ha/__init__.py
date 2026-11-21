"""H4: the Hypothesis Agent as a Contrastive Language Model (docs/HYPOTHESIS_H4.md).

After Contrastive-LM/CLM (Kwok et al., 2026): a **frozen** language model turns the state
(here: the HA prompt with the patient's history and test results) and each action (here: each
of the four diseases) into vectors; two small trainable **projection heads** map them into a
shared space, trained with InfoNCE so that a state lands near its correct action. The
hypothesis is the disease whose vector aligns best; the softmax over the four scores gives the
confidence. The backbone is our own Llama-3-8B-Instruct, either plain (``base``) or with the
frozen H2 HA adapter (``h2``).

- heads.py: the projection heads, the loss, saving and loading (pure torch)
- backbone.py: last-token pooled embeddings from the (frozen) Llama
- embed.py: stage 1a, embed all H2 training/validation records and the fixed test records
- train_heads.py: stage 1b, train the heads on cached embeddings, choose on validation, score test
- scorer.py: the HA turn for the environment (the existing FrozenHAEnvironment is reused)
- evaluate.py: stage 3 and 4 test runs (a DA adapter + the contrastive HA)
- train_joint.py: stage 4, a new DA trained while the heads keep learning (or stay frozen)

Nothing outside this package is changed except new config, slurm and test files.
"""
