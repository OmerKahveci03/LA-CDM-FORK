# H4b: the baseline DA with the contrastive HA vs with the H2 HA (step 2) and vs the baseline model.
#   sbatch --export=ALL,SCRIPT=jobs/contrastive_h4b_stats.sh,B=base slurm/cpu_script.sbatch
# Output: outputs/contrastive/stats/h4b_<score>/
: "${B:?Set B=base or B=h2}"
R=outputs/cost_sweep/cost_1_attempt2
for score in diagnosis final_hypothesis; do
  "$PY" -u scripts/test_stats.py --score "$score" --resamples 2000 \
    --focus-dir "contrastive=outputs/contrastive/combined/$B" \
    --extra "step2_h2=$R/frozen_ha_seed0_step_0600" --extra "baseline=$R/test_step_0600" \
    --out-dir "outputs/contrastive/stats/h4b_$score"
done
