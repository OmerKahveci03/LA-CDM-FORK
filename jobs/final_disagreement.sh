# A1 (docs/FINAL_ANALYSES.md): DA-HA disagreement and deferral rules on the MIMIC test set.
#   sbatch --export=ALL,SCRIPT=jobs/final_disagreement.sh slurm/cpu_script.sbatch
# Output: outputs/final_analyses/a1_disagreement.{md,json}. Rerun after A4 finishes to add those rows.
"$PY" -u scripts/final_disagreement.py --out-dir outputs/final_analyses
