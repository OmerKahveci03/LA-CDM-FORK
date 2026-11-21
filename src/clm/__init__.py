"""H4: the Hypothesis Agent as a Context Language Model (docs/HYPOTHESIS_H4.md, docs/CLM_PRIMER.md).

The HA keeps its own context (the patient information and test results so far) and may
rewrite it into notes at every turn. Its hypothesis is read from that context by four-way
scoring (src/ha/scoring.py), exactly as the H2 HA is scored.

- context.py: pure Python (context state, rewrite parsing, advantages, record sequences)
- agent.py: the model side (generation of rewrites, scoring, training mode)
- learning.py: one HA update (stepwise GRPO for the rewrites + SFT for the diagnosis)
- replay.py: the HA on fixed record sequences (stage 2 and test scoring of stage 3)
- train_ha.py: stage 3, the CLM HA trained alone
- environment.py, train_joint.py, evaluate_joint.py: stage 4, the HA and DA trained together

Nothing outside this package is changed except new prompt, config and slurm files.
"""
