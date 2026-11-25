# A6 (docs/FINAL_ANALYSES.md): "H3a + defer" (H3a's HA final hypothesis as the diagnosis) vs the baseline's DA,
# on MIMIC (primary) and Edward's external set; step2 + defer as a secondary, chosen-after-the-fact comparison.
#   sbatch --export=ALL,SCRIPT=jobs/final_defer_vs_baseline.sh slurm/cpu_script.sbatch
# Output: outputs/final_analyses/a6_defer_vs_baseline.{md,json}
"$PY" -u scripts/defer_vs_baseline.py --out-dir outputs/final_analyses
