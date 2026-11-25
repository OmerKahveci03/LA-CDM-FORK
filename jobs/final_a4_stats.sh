# A4a (docs/FINAL_ANALYSES.md): each DA with the HA hidden vs the same DA with the HA shown (paired, 240 patients).
#   sbatch --export=ALL,SCRIPT=jobs/final_a4_stats.sh slurm/cpu_script.sbatch
# Output: outputs/final_analyses/a4a_<da>_<score>/test_stats.md
R=outputs/cost_sweep/cost_1_attempt2
H=outputs/h3/h3a_ha0
for score in diagnosis final_hypothesis; do
  "$PY" -u scripts/test_stats.py --score "$score" --resamples 2000 --focus-dir "hidden=$R/hidden_ha_seed0_step_0600" \
    --extra "baseline_model=$R/test_step_0600" --extra "step2_shown=$R/frozen_ha_seed0_step_0600" \
    --out-dir "outputs/final_analyses/a4a_baseline_$score"
  "$PY" -u scripts/test_stats.py --score "$score" --resamples 2000 --focus-dir "hidden=$H/hidden_ha_seed0_step_0700" \
    --extra "h3a_shown=$H/frozen_ha_seed0_step_0700" \
    --out-dir "outputs/final_analyses/a4a_h3a_$score"
done
