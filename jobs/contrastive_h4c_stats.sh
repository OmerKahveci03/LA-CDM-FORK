# H4c: joint (heads learning) vs frozen heads, each at its validation-chosen step; H3a for reference.
#   sbatch --export=ALL,SCRIPT=jobs/contrastive_h4c_stats.sh,B=base,JSTEP=700,FSTEP=600 slurm/cpu_script.sbatch
# Output: outputs/contrastive/stats/h4c_<score>/
: "${B:?Set B}" "${JSTEP:?Set JSTEP (joint arm step)}" "${FSTEP:?Set FSTEP (frozen arm step)}"
J=outputs/contrastive/joint/joint_$B/test_$(printf 'step_%04d' "$JSTEP")
F=outputs/contrastive/joint/frozen_$B/test_$(printf 'step_%04d' "$FSTEP")
for score in diagnosis final_hypothesis; do
  "$PY" -u scripts/test_stats.py --score "$score" --resamples 2000 --focus-dir "joint=$J" \
    --extra "frozen=$F" --extra "h3a=outputs/h3/h3a_ha0/frozen_ha_seed0_step_0700" \
    --out-dir "outputs/contrastive/stats/h4c_$score"
done
