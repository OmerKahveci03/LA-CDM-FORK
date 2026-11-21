# Contrastive HA (H4): runbook

Every job for H4 (`docs/HYPOTHESIS_H4.md`), in order, with where its output goes. Background:
`docs/CONTRASTIVE_PRIMER.md`.

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
PY=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
$PY -m unittest tests.test_contrastive_ha tests.test_frozen_ha   # tiny model, CPU, seconds
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
for b in base h2; do echo "== $b"; cat outputs/contrastive/heads/$b/summary.json | head -6; done
```

**H4a statistics** for the chosen backbone `B`, against the H2 HA on the same records:
```bash
B=base   # or h2, whichever won on validation
for arm in A_da2 B_none C_random2 D_all; do
  $PY -u scripts/test_stats.py --score diagnosis --resamples 2000 \
    --focus-dir "contrastive_$B=outputs/contrastive/scores/$B/$arm" \
    --extra "h2=outputs/ha_sft/scores/new0/$arm" --extra "baseline_ha=outputs/ha_sft/scores/old2/$arm" \
    --out-dir outputs/contrastive/stats/h4a_$arm
done
```

## Stage 3: the baseline DA with the contrastive HA (H4b)

```bash
sbatch --export=ALL,BACKBONE=$B,SMOKE=1 slurm/contrastive_evaluate.sbatch   # small sample first
sbatch --export=ALL,BACKBONE=$B slurm/contrastive_evaluate.sbatch           # -> outputs/contrastive/combined/$B/
R=outputs/cost_sweep/cost_1_attempt2
for score in diagnosis final_hypothesis; do
  $PY -u scripts/test_stats.py --score $score --resamples 2000 \
    --focus-dir "contrastive=outputs/contrastive/combined/$B" \
    --extra "step2_h2=$R/frozen_ha_seed0_step_0600" --extra "baseline=$R/test_step_0600" \
    --out-dir outputs/contrastive/stats/h4b_$score
done
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
- **Choose each arm's step on validation:**
  `$PY scripts/validation_curve.py outputs/contrastive/joint/<arm>_$B/train`. Ignore the
  "cycle" labels: every step is a DA step.
- **Test** each arm at its chosen step:
  ```bash
  sbatch --export=ALL,BACKBONE=$B,RUN=joint_$B,ADAPTER_STEP=<best>  slurm/contrastive_evaluate.sbatch
  sbatch --export=ALL,BACKBONE=$B,RUN=frozen_$B,ADAPTER_STEP=<best> slurm/contrastive_evaluate.sbatch
  ```
- **H4c statistics:**
  ```bash
  J=outputs/contrastive/joint/joint_$B/test_step_XXXX; F=outputs/contrastive/joint/frozen_$B/test_step_YYYY
  for score in diagnosis final_hypothesis; do
    $PY -u scripts/test_stats.py --score $score --resamples 2000 --focus-dir "joint=$J" \
      --extra "frozen=$F" --extra "h3a=outputs/h3/h3a_ha0/frozen_ha_seed0_step_0700" \
      --out-dir outputs/contrastive/stats/h4c_$score
  done
  ```

## What each job prints that matters

| Job | Look for |
|---|---|
| embed | `[embed train] N/7676 (…s)`, then `[embed] done` |
| heads | `[heads] epoch k: train acc … val acc …`, `[heads] chosen epoch …`, `[heads] test <arm>: accuracy …` |
| evaluate | `[contrastive HA] DA …; backbone …; heads …`, then the usual test metrics |
| train_joint | `[contrastive HA] backbone … (frozen), heads … (… parameters, updated every step)` |
