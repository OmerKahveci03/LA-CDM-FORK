# A3 and A2 (docs/FINAL_ANALYSES.md): combined HAs (seconds) and the CLM head ensemble with
# confidence bounds (10 members on CPU, about 30-60 minutes).
#   sbatch --export=ALL,SCRIPT=jobs/final_ha_offline.sh slurm/cpu_script.sbatch
# Output: outputs/final_analyses/a3_combined_ha.{md,json}, a2_ensemble_bounds.{md,json}, a2_ensemble_probs_*.csv
"$PY" -u scripts/combined_ha.py --out-dir outputs/final_analyses
"$PY" -u -m src.contrastive_ha.ensemble_bounds --emb-dir outputs/contrastive/embeddings/h2 --out-dir outputs/final_analyses
