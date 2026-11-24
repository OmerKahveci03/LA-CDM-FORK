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

## Files

| File | Origin |
|---|---|
| `trained_ext_v2.sbatch` | Edward's `slurm/ext_v2/trained_ext_v2.sbatch`, with the same evaluation calls. Adapted to run from this repo, logs in `logs/`, 24G memory, and the final `chgrp` limited to `outputs/ext_v2_trained`. His version also changed the group of the whole `logs/` folder, which holds MIMIC-derived text. |
| `da_ha_disagreement.py` | Edward's DA–HA disagreement table (`NEXT_EXPERIMENTS.md` item 1), copied **unchanged** and reviewed: standard library only; reads one episodes file and writes one JSON. Its confidence thresholds 0–10 match the 0–10 scale our episode files store. Its threshold sweep is descriptive; any threshold must be chosen on MIMIC validation episodes. |

The disagreement script also works on our own MIMIC test and validation episodes, for the Wednesday
analysis: `$PY edward/da_ha_disagreement.py <episodes csv> --name ... --out ...` (run it inside a CPU job).
