# H4c extension (exploratory, docs/HYPOTHESIS_H4.md): the frozen arm at its validation-chosen step over
# 100-1900 (default 1100) against the same arm at 400 (the pre-registered H4c choice), H3a and the baseline.
#   sbatch --export=ALL,SCRIPT=jobs/contrastive_h4c_ext_stats.sh slurm/cpu_script.sbatch
#   optional: FSTEP=1100
# Output: outputs/contrastive/stats/h4c_ext_<score>/
FSTEP="${FSTEP:-1100}"
F=outputs/contrastive/joint/frozen_h2/test_$(printf 'step_%04d' "$FSTEP")
R=outputs/cost_sweep/cost_1_attempt2
for score in diagnosis final_hypothesis; do
  "$PY" -u scripts/test_stats.py --score "$score" --resamples 2000 --focus-dir "frozen_$FSTEP=$F" \
    --extra "frozen_400=outputs/contrastive/joint/frozen_h2/test_step_0400" \
    --extra "h3a=outputs/h3/h3a_ha0/frozen_ha_seed0_step_0700" --extra "baseline=$R/test_step_0600" \
    --out-dir "outputs/contrastive/stats/h4c_ext_$score"
done
