# H4a (docs/HYPOTHESIS_H4.md): contrastive HA vs the H2 HA and the baseline HA on the same records.
#   sbatch --export=ALL,SCRIPT=jobs/contrastive_h4a_stats.sh,B=base slurm/cpu_script.sbatch
# B = the backbone chosen on validation (base or h2). Output: outputs/contrastive/stats/h4a_<arm>/
: "${B:?Set B=base or B=h2}"
for b in base h2; do
  if [[ -f outputs/contrastive/heads/$b/summary.json ]]; then
    echo "== heads $b"; head -6 "outputs/contrastive/heads/$b/summary.json"
  fi
done
for arm in A_da2 B_none C_random2 D_all; do
  "$PY" -u scripts/test_stats.py --score diagnosis --resamples 2000 \
    --focus-dir "contrastive_$B=outputs/contrastive/scores/$B/$arm" \
    --extra "h2=outputs/ha_sft/scores/new0/$arm" --extra "baseline_ha=outputs/ha_sft/scores/old2/$arm" \
    --out-dir "outputs/contrastive/stats/h4a_$arm"
done
