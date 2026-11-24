# H4 extension restart: where we are and what to do next

**Written November 23, 2026 (evening).** Background: `docs/HYPOTHESIS_H4.md`, section 8,
"H4c-ext". Everything else in H4 (H4a, H4b, H4c at 700 steps) is finished and is not affected.

## What happened

1. We extended both stage-4 arms (joint, frozen) from 700 to 1,900 steps (exploratory).
2. The resume at step 700 silently reset the DA to an untrained adapter. `transformers.Trainer`
   reloads only the adapter in the checkpoint's `contrastive_backbone/` subfolder, never the
   DA's root adapter. You can see it in `figures/h4/h4_end_reasons`: invalid formats jump to
   about 90% right after step 700. The same bug explains H3's late "instability" (the
   correction is in `docs/HYPOTHESIS_H3.md`).
3. **Fix:** `src/utils/peft_resume.py` (`RestoreAdapterCallback`), used in
   `src/contrastive_ha/train_joint.py`. Test: `tests/test_peft_resume.py`.
4. `save_total_limit: 5` had already deleted `checkpoint-700`, so we rebuild it from the
   `adapters/step_0700` snapshot (`scripts/rebuild_checkpoint.py`). The DA weights, heads and
   data position are exact. **The AdamW moments restart fresh** (documented in `HYPOTHESIS_H4.md`).

## Status when this was written

| Job | What | State |
|---|---|---|
| 52514777 | unit tests (`tests.test_peft_resume tests.test_contrastive_ha`) | queued |
| 52514778 | rebuild `checkpoint-700` for both arms and move the post-700 outputs to `train/broken_resume_after_700/` (`jobs/rebuild_h4_checkpoint_700.sh`) | waits for the tests (`afterok`) |
| — | the broken extension jobs | cancelled |

## Next steps (in order)

**1. Check the tests and the rebuild** (`cat`/`tail` only):
```bash
cd /blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
tail -n 5 logs/lacdm_tests_52514777.out     # must end with OK
tail -n 20 logs/lacdm_cpu_52514778.out      # per arm: "Rebuilt .../checkpoint-700", "Moved N items after step 700"
ls outputs/contrastive/joint/joint_h2/train/checkpoint-700 outputs/contrastive/joint/frozen_h2/train/checkpoint-700
```
Each `checkpoint-700` should contain `adapter_config.json`, `adapter_model.safetensors`,
`contrastive_heads.pt`, `trainer_state.json` and `REBUILT.json`.

- If the tests **failed**, the rebuild did not run (nothing was moved). Paste the test log to Claude.
- If job 52514778 shows as **"DependencyNeverSatisfied"** in `squeue`, the tests failed; cancel it with
  `scancel 52514778`.

**2. Restart the extension** (3 chained 24-hour jobs per arm):
```bash
for arm in joint frozen; do
  R=outputs/contrastive/joint/${arm}_h2/train/checkpoint-700
  j=$(sbatch --parsable --export=ALL,ARM=$arm,BACKBONE=h2,MAX_STEPS=1900,RESUME_CHECKPOINT=$R slurm/contrastive_train_joint.sbatch)
  j=$(sbatch --parsable --dependency=afterany:$j --export=ALL,ARM=$arm,BACKBONE=h2,MAX_STEPS=1900,RESUME_CHECKPOINT=$R slurm/contrastive_train_joint.sbatch)
  sbatch --dependency=afterany:$j --export=ALL,ARM=$arm,BACKBONE=h2,MAX_STEPS=1900,RESUME_CHECKPOINT=$R slurm/contrastive_train_joint.sbatch
done
```
`RESUME_CHECKPOINT` is used only while no complete checkpoint exists, so the follow-up jobs
resume from the run's own newer checkpoints.

**3. Check that the resume worked** (about 10 minutes after the jobs start):
```bash
grep -h "Resume:\|\[resume\]" logs/lacdm_con_joint_*.out | tail -4
```
You should see `Resume: .../checkpoint-700  Target: 1900 steps` and
`[resume] restored adapter 'default' ... LoRA norm A -> B` for each arm.

After about an hour:
```bash
tail -n 3 outputs/contrastive/joint/joint_h2/train/contrastive_heads_updates.jsonl   # steps 701+
```
Then rerun the figures job. `h4_end_reasons` must show **no** jump in invalid formats at step 700:
```bash
sbatch --export=ALL,SCRIPT=jobs/figures_h3_h4.sh slurm/cpu_script.sbatch
```

**4. When both arms reach 1,900 steps** (about 2 days): choose each arm's step on validation
over steps 100–1900, test those steps with `slurm/contrastive_evaluate.sbatch` (`RUN=joint_h2` /
`frozen_h2`, `ADAPTER_STEP=...`), and run `jobs/contrastive_h4c_stats.sh` with the new steps.
Report it as exploratory. The 700-step H4c result stays the primary one.

## Open items (not urgent)

- `src/train_frozen_ha.py` (H3) has the same resume bug; nothing uses it right now. The fix is
  the same: add `RestoreAdapterCallback(cfg.training.resume_from_checkpoint)` to its callbacks.
- Untested idea: use the HA's final hypothesis as the diagnosis. In every system the HA
  (87–89%) is as accurate as or more accurate than the DA (82–86%). This is a CPU-only check on
  existing transcripts, with the rule chosen on validation.
- Naming: "separately fine-tuned HA" (was "new HA"), "baseline HA" (jointly trained), "CLM HA".
