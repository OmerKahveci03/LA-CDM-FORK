# Rebuild checkpoint-700 for both H4 arms from adapters/step_0700, then move the broken
# post-700 outputs aside (docs/HYPOTHESIS_H4.md, H4c-ext).
#   sbatch --export=ALL,SCRIPT=jobs/rebuild_h4_checkpoint_700.sh slurm/cpu_script.sbatch
for arm in joint frozen; do
  d=outputs/contrastive/joint/${arm}_h2/train
  "$PY" scripts/rebuild_checkpoint.py "$d" 700
  bash jobs/quarantine_after_step.sh "$d" 700 broken_resume
  ls "$d/checkpoint-700"
done
