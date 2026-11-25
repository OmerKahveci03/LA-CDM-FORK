# H3c (docs/HYPOTHESIS_H3C.md): the pre-registered comparisons for the H3c test run (CPU).
#   sbatch --export=ALL,SCRIPT=jobs/h3c_stats.sh,STEP=<chosen step> slurm/cpu_script.sbatch
# Output: outputs/final_analyses/a6_defer_vs_baseline.{md,json} (with the H3c rows) and
#         outputs/h3/h3c_ha0/stats_<score>/test_stats.md (cost and tests vs H3a and the baseline)
: "${STEP:?Set STEP (the validation-chosen H3c step)}"
D=outputs/h3/h3c_ha0/defer_ha_seed0_$(printf 'step_%04d' "$STEP")
"$PY" -u scripts/defer_vs_baseline.py --out-dir outputs/final_analyses --h3c-dir "$D"
for score in diagnosis final_hypothesis; do
  "$PY" -u scripts/test_stats.py --score "$score" --resamples 2000 --focus-dir "h3c=$D" \
    --extra "h3a=outputs/h3/h3a_ha0/frozen_ha_seed0_step_0700" \
    --extra "baseline=outputs/cost_sweep/cost_1_attempt2/test_step_0600" \
    --out-dir "outputs/h3/h3c_ha0/stats_$score"
done
