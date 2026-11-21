# CLM Hypothesis Agent (H4): runbook

Every job for H4 (`docs/HYPOTHESIS_H4.md`), in order, with where its output goes. Background:
`docs/CLM_PRIMER.md`.

All output is under **`outputs/clm/`**. Nothing outside it is written, and nothing existing
is changed.

```
outputs/clm/
├── replay/<name>/<arm>/            stage 2 and stage-3 test scoring (fixed test sequences)
├── replay_smoke/<name>/<arm>/      the same, 5 patients (smoke tests)
├── ha_train/<init>init_seed0/      stage 3: the CLM HA trained alone
│   ├── adapter/                    best adapter by validation (written at the end)
│   ├── adapters/step_XXXX/         adapter at every validation
│   ├── val/step_XXXX/              validation notes + summary; val_curve.csv summarises them
│   ├── checkpoint/                 newest save (automatic resume)
│   └── updates.jsonl               one line of training statistics per step
├── ha_train/smoke_<init>/          stage-3 smoke test
└── joint/<arm>_<ha_run>/           stage 4: arm = joint (HA keeps learning) or frozen
    ├── train/                      like any training run (episodes/, adapters/step_XXXX/,
    │                               checkpoints); the HA adapter of each snapshot is in
    │                               adapters/step_XXXX/clm_ha; clm_ha_updates.jsonl; clm_notes.jsonl
    └── test_step_XXXX/             test evaluation of the chosen step (+ clm_notes.jsonl)
```

## 0. Before anything

```bash
cd /blue/prismap-ai-core/omerkahveci/LA-CDM-FORK && git pull
PY=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
$PY -m unittest tests.test_clm tests.test_frozen_ha    # the CLM tests use a tiny model; no GPU
```

## Stage 2 (H4a): the CLM HA out of the box (evaluation only)

**Smoke tests first** (5 patients per arm, a few minutes each):
```bash
sbatch --export=ALL,MODEL=base,MODE=edit,SMOKE=1 slurm/clm_replay.sbatch
sbatch --export=ALL,MODEL=h2,MODE=edit,SMOKE=1   slurm/clm_replay.sbatch
```
To check one: `cat outputs/clm/replay_smoke/base_edit/overview.json`, then read a few notes with
`head -c 3000 outputs/clm/replay_smoke/base_edit/D_all/notes.jsonl`.

**Full runs** (4 jobs; they can run in parallel; edit mode takes a few hours, append-only less):
```bash
sbatch --export=ALL,MODEL=base,MODE=edit   slurm/clm_replay.sbatch   # -> outputs/clm/replay/base_edit/
sbatch --export=ALL,MODEL=base,MODE=append slurm/clm_replay.sbatch   # -> outputs/clm/replay/base_append/
sbatch --export=ALL,MODEL=h2,MODE=edit     slurm/clm_replay.sbatch   # -> outputs/clm/replay/h2_edit/
sbatch --export=ALL,MODEL=h2,MODE=append   slurm/clm_replay.sbatch   # -> outputs/clm/replay/h2_append/
```
`h2_append` must reproduce the H2 scores (`outputs/ha_sft/scores/new0/`), which is a check
that the replay is correct.

**Analysis** (paste the output):
```bash
R=outputs/clm/replay
for n in base_edit base_append h2_edit h2_append; do echo "== $n"; $PY -c "
import json,sys; o=json.load(open(sys.argv[1]))
for a,s in o.items(): print(f\"{a:10s} acc {s['accuracy']:.3f}  rewrite {s['rewrite_rate']:.2f}  malformed {s['malformed_rate']:.2f}  final ctx {s['mean_final_context_tokens']:.0f}  total ctx {s['mean_total_context_tokens']:.0f}\")
" $R/$n/overview.json; done
for m in base h2; do for arm in D_all A_da2; do
  $PY -u scripts/test_stats.py --score diagnosis --resamples 2000 \
    --focus-dir "${m}_edit=$R/${m}_edit/$arm" --extra "${m}_append=$R/${m}_append/$arm" \
    --out-dir outputs/clm/stats/stage2_${m}_$arm
done; done
```
In these rows "diagnosis" is the HA's final hypothesis, so `--score diagnosis` compares the HAs.
The rule (H4 section 5) applies to base, D_all. The h2 comparison decides INIT for stage 3
(section 7a).

## Stage 3 (H4b): train the CLM HA alone

```bash
sbatch --export=ALL,INIT=h2,SMOKE=1 slurm/clm_train_ha.sbatch   # 4 steps -> outputs/clm/ha_train/smoke_h2/
sbatch --export=ALL,INIT=h2 slurm/clm_train_ha.sbatch           # 600 steps -> outputs/clm/ha_train/h2init_seed0/
```
- Use `INIT=base` instead if stage 2 says so.
- It probably needs about 1.5 jobs of 24 hours: resubmit the same command after the time-out
  and it resumes from `checkpoint/`.
- **Progress:** `tail -3 outputs/clm/ha_train/h2init_seed0/updates.jsonl` and
  `cat outputs/clm/ha_train/h2init_seed0/val_curve.csv`.
- When finished, `training_complete.json` names the best step, chosen on validation only.

**Test scoring of the trained CLM HA** (same sequences as stage 2):
```bash
sbatch --export=ALL,MODEL=stage3,HA_RUN=h2init_seed0,MODE=edit slurm/clm_replay.sbatch
# -> outputs/clm/replay/stage3-h2init_seed0_edit/
for arm in A_da2 C_random2 D_all B_none; do
  $PY -u scripts/test_stats.py --score diagnosis --resamples 2000 \
    --focus-dir "clm=outputs/clm/replay/stage3-h2init_seed0_edit/$arm" \
    --extra "h2=outputs/ha_sft/scores/new0/$arm" --out-dir outputs/clm/stats/stage3_$arm
done
```

## Stage 4 (H4c): the DA and the CLM HA together

```bash
sbatch --export=ALL,ARM=joint,HA_RUN=h2init_seed0,SMOKE=1 slurm/clm_train_joint.sbatch  # 4 steps
sbatch --export=ALL,ARM=joint,HA_RUN=h2init_seed0  slurm/clm_train_joint.sbatch          # 700 steps
sbatch --export=ALL,ARM=frozen,HA_RUN=h2init_seed0 slurm/clm_train_joint.sbatch          # optional arm
```
- Resubmit after each 24-hour time-out; it resumes, including the HA's weights and optimiser.
- To queue the follow-up job: `sbatch --dependency=afterany:<jobid> ...` (same command).
- **Choose the checkpoint on validation:**
  `$PY scripts/validation_curve.py outputs/clm/joint/joint_h2init_seed0/train`. Ignore its
  "cycle" labels: every step here is a DA step.
- **Test** the chosen step:
  ```bash
  sbatch --export=ALL,RUN=joint_h2init_seed0,ADAPTER_STEP=<best> slurm/clm_evaluate_joint.sbatch
  # -> outputs/clm/joint/joint_h2init_seed0/test_step_XXXX/
  ```
- **Primary comparison:** joint minus H3a (paired, the DA's diagnosis), plus the final hypothesis:
  ```bash
  J=outputs/clm/joint/joint_h2init_seed0/test_step_XXXX
  for score in diagnosis final_hypothesis; do
    $PY -u scripts/test_stats.py --score $score --resamples 2000 --focus-dir "clm_joint=$J" \
      --extra "h3a=outputs/h3/h3a_ha0/frozen_ha_seed0_step_0700" \
      --extra "baseline=outputs/cost_sweep/cost_1_attempt2/test_step_0600" \
      --out-dir outputs/clm/stats/stage4_$score
  done
  ```

## What each job prints that matters

| Job | Look for |
|---|---|
| replay | `[clm replay] <arm>: accuracy …, rewrite rate …, final context … tokens` |
| train_ha | `[clm train_ha] val step N: accuracy …` and `step N: success …, rewrite rate …` |
| train_joint | `[clm joint] CLM HA … as adapter 'clm_ha' (… weights, updated every step)`, then the usual validation dictionaries |
