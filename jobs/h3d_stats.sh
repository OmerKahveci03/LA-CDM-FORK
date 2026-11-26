# H3d (docs/HYPOTHESIS_H3D.md): the pre-registered comparisons for the H3d test run (CPU).
#   sbatch --export=ALL,SCRIPT=jobs/h3d_stats.sh,DESIGN=<defer|plain>,STEP=<chosen step> slurm/cpu_script.sbatch
# Primary: H3d vs the same design at cost weight 1 (H3c step 500 for defer; H3a step 700 for plain).
# Output: outputs/h3/h3d_<design>_ha0/stats_<score>/test_stats.md and outputs/final_analyses/a6_defer_vs_baseline.md
: "${DESIGN:?Set DESIGN=defer or plain}" "${STEP:?Set STEP (the validation-chosen H3d step)}"
TAG=$(printf 'step_%04d' "$STEP")
H3A=outputs/h3/h3a_ha0/frozen_ha_seed0_step_0700
if [[ "$DESIGN" == "defer" ]]; then
  D=outputs/h3/h3d_defer_ha0/defer_ha_seed0_$TAG
  SAME="h3c_weight1=outputs/h3/h3c_ha0/defer_ha_seed0_step_0500"
else
  D=outputs/h3/h3d_plain_ha0/frozen_ha_seed0_$TAG
  SAME="h3a_weight1=$H3A"
fi
for score in diagnosis final_hypothesis; do
  "$PY" -u scripts/test_stats.py --score "$score" --resamples 2000 --focus-dir "h3d_$DESIGN=$D" \
    --extra "$SAME" --extra "h3a=$H3A" --extra "baseline=outputs/cost_sweep/cost_1_attempt2/test_step_0600" \
    --out-dir "outputs/h3/h3d_${DESIGN}_ha0/stats_$score"
done
"$PY" -u scripts/defer_vs_baseline.py --out-dir outputs/final_analyses --h3d-dir "$D" \
  --h3c-dir outputs/h3/h3c_ha0/defer_ha_seed0_step_0500
