# A5 (docs/FINAL_ANALYSES.md): every system against the baseline model on the same 240 MIMIC test patients,
# paired (exact McNemar for accuracy, paired bootstrap 95% CIs, permutation tests, Holm across the systems).
#   sbatch --export=ALL,SCRIPT=jobs/final_vs_baseline_stats.sh slurm/cpu_script.sbatch
# Output: outputs/final_analyses/a5_vs_baseline_<score>/test_stats.md ("baseline minus each other run")
R=outputs/cost_sweep/cost_1_attempt2
J=outputs/contrastive/joint
for score in diagnosis final_hypothesis; do
  "$PY" -u scripts/test_stats.py --score "$score" --resamples 2000 --focus-dir "baseline=$R/test_step_0600" \
    --extra "step2=$R/frozen_ha_seed0_step_0600" \
    --extra "h3a=outputs/h3/h3a_ha0/frozen_ha_seed0_step_0700" \
    --extra "h3b=outputs/h3/h3b_ha0/frozen_ha_seed0_step_0700" \
    --extra "h4b=outputs/contrastive/combined/h2" \
    --extra "h4c_joint=$J/joint_h2/test_step_0600" \
    --extra "h4c_frozen=$J/frozen_h2/test_step_0400" \
    --out-dir "outputs/final_analyses/a5_vs_baseline_$score"
done
