# External validation set v2 (from Edward Kemp)

Code and instructions from teammate Edward Kemp, copied into this repo on November 24, 2026, so the
external evaluation runs from here instead of from his folder.

## The data (not MIMIC)

- **Source:** 152 patients from audited, CC BY-licensed published case reports, rewritten into the
  MIMIC-CDM CSV format. The `Patient History Summary` column was produced by our history summariser
  (his job 52542556), in the same way as for MIMIC-CDM.
- **Files** (on HiPerGator; `data/` is git-ignored, so they are not committed):
  - `data/ext_v2/ext_v2_test_strict_l3sum.csv`: the test set used (152 rows)
  - `data/ext_v2/ext_v2_lab_test_mapping.csv`: the lab test mapping used
  - `data/ext_v2/ext_v2_test_strict.csv`: the same patients with a different history summary (unused)
- **Original location:** `/blue/cai6734/kempaedward/LA-CDM-FORK/data/`
- **Labels:** appendicitis 56, pancreatitis 55, cholecystitis 27, diverticulitis 14.

**What to keep in mind** (checked November 24, 2026):
- Radiology reports name the true disease for 56/152 patients (appendicitis 34/56,
  cholecystitis 15/27, diverticulitis 9/14, pancreatitis 0/55). MIMIC impressions often do this
  too, but imaging-based cases are easier here.
- 6 LLM-written history summaries name the diagnosis ("suggest a possible case of acute
  appendicitis"), which is minor leakage.
- This is a different population from MIMIC: unusual cases (case reports), 12 patients under 18,
  shorter records, and a different class balance. With 152 patients, intervals are about ±6–8 points.
- **Evaluation only:** every checkpoint was fixed beforehand on MIMIC validation. Never tune
  or choose anything on this set.

## Copy the data (once, file commands only)

```bash
cd /blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
mkdir -p data/ext_v2
cp /blue/cai6734/kempaedward/LA-CDM-FORK/data/ext_v2_test_strict_l3sum.csv \
   /blue/cai6734/kempaedward/LA-CDM-FORK/data/ext_v2_lab_test_mapping.csv \
   /blue/cai6734/kempaedward/LA-CDM-FORK/data/ext_v2_test_strict.csv data/ext_v2/
```

## Run

```bash
sbatch edward/trained_ext_v2.sbatch                        # baseline, h3a, h4b, h4c (about 1 hour each)
sbatch --export=ALL,ARMS=baseline edward/trained_ext_v2.sbatch
```
- Log: `logs/lacdm_ext_v2_<jobid>.out`
- Results: `outputs/ext_v2_trained/<arm>/`
- Share with Edward: `cp -r outputs/ext_v2_trained /blue/cai6734/omerkahveci/`. This holds only
  results on his data, with no MIMIC and no model weights.

## Results (job 52653909, November 24, 2026; commit 64baf59)

All four arms finished (about 30 minutes each). Statistics: `jobs/ext_v2_stats.sh` →
`outputs/ext_v2_trained/stats_{diagnosis,final_hypothesis}/test_stats.md`. The results were shared with
Edward at `/blue/cai6734/omerkahveci/ext_v2_trained/`.

| System | DA accuracy, external [95% CI] | HA final hypothesis, external | Cost/case, external | DA accuracy, MIMIC test |
|---|---|---|---|---|
| Baseline (attempt 2, step 600) | 86.8% [80.5, 91.3] | 90.1% | $1,374 | 85.4% |
| H3a (retrained DA + separately fine-tuned HA) | 89.5% [83.6, 93.4] | 92.1% | $1,077 | 85.8% |
| H4b (baseline DA + CLM HA) | 87.5% [81.3, 91.8] | 90.1% | $1,320 | 82.9% |
| H4c (joint DA + CLM HA, step 600) | 78.9% [71.8, 84.7] | 82.2% | $1,181 | 84.2% |

Paired, the baseline minus each system (same 152 patients; exact McNemar, with Holm correction over the 3 comparisons):
- **H3a:** accuracy H3a +2.6 points [−3.3, +7.9] (12 vs 8 discordant), not significant;
  **cost H3a −$298 [−$401, −$203], p < 0.001**. H3a's ~20% cost saving **replicates on
  independent data**. This is also the first direct test of H3a against the baseline model.
- **H4b:** accuracy −0.7 [−5.9, +4.6], cost +$55 (p = 0.21); no difference.
- **H4c:** accuracy H4c −7.9 points [−15.1, −1.3] (10 vs 22), McNemar p = 0.050, **Holm p = 0.15**
  (a likely drop, not established); cost H4c −$193 [−$304, −$91]. Its HA also dropped (82.2% vs
  90.1% for the same CLM HA in H4b). A hypothesis, not verified: the continuously updated heads
  fitted MIMIC-style records and generalise worse.

DA–HA disagreement (`da_ha_disagreement.json`): in every system the DA overrode a correct HA at
least as often as it corrected a wrong one (overrode/corrected: baseline 3/1 plus 3 undiagnosed with
the HA right; H3a 4/0; H4b 2/2 plus 4 undiagnosed with the HA right; H4c 8/3). So "always use the
HA's final hypothesis" was 2.6–3.3 points better than the DA in every system. The counts are small and
not significant on their own (for example 4 vs 0, p ≈ 0.13), but the direction matches MIMIC. That
rule has no parameter and was written down before these results (`docs/NEXT_EXPERIMENTS.md` item 1),
so it can be reported here; any confidence threshold must be chosen on MIMIC validation, not on this set.

How to read it: this set is probably easier than MIMIC (radiology names the answer for 37% of
patients), so the fair claim is "generalises at least as well", not "does better".

## Files

| File | Origin |
|---|---|
| `trained_ext_v2.sbatch` | Edward's `slurm/ext_v2/trained_ext_v2.sbatch`, with the same evaluation calls. Adapted to run from this repo, logs in `logs/`, 24G memory, and the final `chgrp` limited to `outputs/ext_v2_trained`. His version also changed the group of the whole `logs/` folder, which holds MIMIC-derived text. |
| `da_ha_disagreement.py` | Edward's DA–HA disagreement table (`NEXT_EXPERIMENTS.md` item 1), copied **unchanged** and reviewed: standard library only; reads one episodes file and writes one JSON. Its confidence thresholds 0–10 match the 0–10 scale our episode files store. Its threshold sweep is descriptive; any threshold must be chosen on MIMIC validation episodes. |

The disagreement script also works on our own MIMIC test and validation episodes, for the Wednesday
analysis: `$PY edward/da_ha_disagreement.py <episodes csv> --name ... --out ...` (run it inside a CPU job).
