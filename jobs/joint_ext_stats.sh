# H4c extension, joint arm (exploratory, docs/HYPOTHESIS_H4.md): the rerun's step 1000 (the best validation step
# with >= 0.5 tests per patient) against the original joint step 600, the frozen arm's step 1100, H3a and the baseline.
#   sbatch --export=ALL,SCRIPT=jobs/joint_ext_stats.sh slurm/cpu_script.sbatch
# Output: outputs/contrastive/stats/joint_ext_<score>/
J=outputs/contrastive/joint
for score in diagnosis final_hypothesis; do
  "$PY" -u scripts/test_stats.py --score "$score" --resamples 2000 --focus-dir "joint_1000=$J/joint_h2/test_step_1000" \
    --extra "joint_600_original=$J/joint_h2/test_step_0600" --extra "frozen_1100=$J/frozen_h2/test_step_1100" \
    --extra "h3a=outputs/h3/h3a_ha0/frozen_ha_seed0_step_0700" \
    --extra "baseline=outputs/cost_sweep/cost_1_attempt2/test_step_0600" \
    --out-dir "outputs/contrastive/stats/joint_ext_$score"
done
