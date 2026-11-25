# Statistics for Edward's external test set v2 (edward/README.md): the four fixed systems on the same 152
# patients, with the same intervals and paired tests as our MIMIC results. Evaluation only.
#   sbatch --export=ALL,SCRIPT=jobs/ext_v2_stats.sh slurm/cpu_script.sbatch
# Output: outputs/ext_v2_trained/stats_<score>/test_stats.md (the "vs the paper" table there refers to MIMIC
# and does not apply to this data set).
E=outputs/ext_v2_trained
for score in diagnosis final_hypothesis; do
  "$PY" -u scripts/test_stats.py --score "$score" --resamples 2000 \
    --focus-dir "baseline=$E/baseline" --extra "h3a=$E/h3a" --extra "h4b=$E/h4b" --extra "h4c=$E/h4c" \
    --out-dir "$E/stats_$score"
done
