# Contrastive HA (H4): runbook

Every job for H4 (`docs/HYPOTHESIS_H4.md`), in order, with where its output goes. Background:
`docs/CONTRASTIVE_PRIMER.md`.

> **In progress (2026-11-23):** the 1,900-step extension is being restarted after a resume
> bug. Current state and next commands: **`docs/H4_EXTENSION_RESTART.md`**.

All output is under **`outputs/contrastive/`**. Nothing else is written or changed.

```
outputs/contrastive/
├── embeddings/<backbone>/          stage 1a: train.pt, val.pt, actions.pt, test/<arm>.pt, meta.json
├── heads/<backbone>/               stage 1b: heads.pt (the chosen epoch), curve.csv, summary.json
├── scores/<backbone>/<arm>/        stage 1b: test scores of the HA alone (same format as H2 scores)
├── combined/<backbone>/            stage 3: baseline DA + contrastive HA, test run
├── joint/<arm>_<backbone>/         stage 4: arm = joint (heads learn) or frozen
│   ├── train/                      DA training run; heads saved in adapters/step_XXXX/contrastive_heads.pt;
│   │                               contrastive_heads_updates.jsonl logs every heads update
│   └── test_step_XXXX/             test run of the chosen step
├── stats/                          scripts/test_stats.py outputs
└── *_smoke/                        smoke-test versions of the above
```

`<backbone>` is `base` (Llama-3-8B-Instruct, no adapter) or `h2` (the same with the frozen H2
HA adapter).

## 0. Before anything

```bash
cd /blue/prismap-ai-core/omerkahveci/LA-CDM-FORK && git pull
sbatch --export=ALL,TESTS="tests.test_contrastive_ha tests.test_frozen_ha" slurm/unit_tests.sbatch
# -> logs/lacdm_tests_<jobid>.out ends with OK (never run Python on the login node)
```

## Stage 1: embed, train the heads, choose the backbone (H4a)

**Smoke test** (8 patients per split; minutes):
```bash
sbatch --export=ALL,BACKBONE=base,SMOKE=1 slurm/contrastive_embed.sbatch
# when it is done:
sbatch --export=ALL,BACKBONE=base,SMOKE=1 slurm/contrastive_heads.sbatch
```

**Full runs.** The two embedding jobs can run in parallel. They take a few hours each and are
resumable (resubmit the same command).
```bash
sbatch --export=ALL,BACKBONE=base slurm/contrastive_embed.sbatch    # -> outputs/contrastive/embeddings/base/
sbatch --export=ALL,BACKBONE=h2   slurm/contrastive_embed.sbatch    # -> outputs/contrastive/embeddings/h2/
# when each finishes (minutes):
sbatch --export=ALL,BACKBONE=base slurm/contrastive_heads.sbatch    # -> heads/base/, scores/base/
sbatch --export=ALL,BACKBONE=h2   slurm/contrastive_heads.sbatch    # -> heads/h2/,   scores/h2/
```

**Choose the backbone (validation only):**
```bash
for b in base h2; do echo "== $b"; head -6 outputs/contrastive/heads/$b/summary.json; done
```

**H4a statistics** for the chosen backbone, on a compute node (CPU job):
```bash
B=base   # or h2, whichever won on validation
sbatch --export=ALL,SCRIPT=jobs/contrastive_h4a_stats.sh,B=$B slurm/cpu_script.sbatch
# -> outputs/contrastive/stats/h4a_<arm>/test_stats.md, log in logs/lacdm_cpu_<jobid>.out
```

## Stage 3: the baseline DA with the contrastive HA (H4b)

```bash
sbatch --export=ALL,BACKBONE=$B,SMOKE=1 slurm/contrastive_evaluate.sbatch   # small sample first
sbatch --export=ALL,BACKBONE=$B slurm/contrastive_evaluate.sbatch           # -> outputs/contrastive/combined/$B/
sbatch --export=ALL,SCRIPT=jobs/contrastive_h4b_stats.sh,B=$B slurm/cpu_script.sbatch
```

## Stage 4: a new DA, the heads learning or frozen (H4c)

```bash
sbatch --export=ALL,ARM=joint,BACKBONE=$B,SMOKE=1 slurm/contrastive_train_joint.sbatch   # 4 steps
sbatch --export=ALL,ARM=joint,BACKBONE=$B  slurm/contrastive_train_joint.sbatch          # 700 steps
sbatch --export=ALL,ARM=frozen,BACKBONE=$B slurm/contrastive_train_joint.sbatch          # 700 steps
```
- Each job stops after 24 hours. Resubmit the same command, or queue it in advance with
  `sbatch --dependency=afterany:<jobid> ...`. A run resumes with the DA, the heads and both
  optimisers.
- **Progress:** `tail -3 outputs/contrastive/joint/joint_$B/train/contrastive_heads_updates.jsonl`
- **Choose each arm's step on validation:** read
  `outputs/contrastive/joint/<arm>_$B/train/episodes/val_step_*_summary.json`, or ask me to
  summarise them. Every step is a DA step.
- **Test** each arm at its chosen step:
  ```bash
  sbatch --export=ALL,BACKBONE=$B,RUN=joint_$B,ADAPTER_STEP=<best>  slurm/contrastive_evaluate.sbatch
  sbatch --export=ALL,BACKBONE=$B,RUN=frozen_$B,ADAPTER_STEP=<best> slurm/contrastive_evaluate.sbatch
  ```
- **H4c statistics** (CPU job; the steps are each arm's validation choice):
  ```bash
  sbatch --export=ALL,SCRIPT=jobs/contrastive_h4c_stats.sh,B=$B,JSTEP=<joint step>,FSTEP=<frozen step> slurm/cpu_script.sbatch
  ```

## What each job prints that matters

| Job | Look for |
|---|---|
| embed | `[embed train] N/7676 (…s)`, then `[embed] done` |
| heads | `[heads] epoch k: train acc … val acc …`, `[heads] chosen epoch …`, `[heads] test <arm>: accuracy …` |
| evaluate | `[contrastive HA] DA …; backbone …; heads …`, then the usual test metrics |
| train_joint | `[contrastive HA] backbone … (frozen), heads … (… parameters, updated every step)` |
