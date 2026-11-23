# Figures for H3 and H4 (docs/HYPOTHESIS_H3.md, docs/HYPOTHESIS_H4.md).
#   sbatch --export=ALL,SCRIPT=jobs/figures_h3_h4.sh slurm/cpu_script.sbatch
#   optional: JSTEP / FSTEP = the joint / frozen arm's validation-chosen step (default 600 / 400)
# Output: figures/h3/ and figures/h4/ (.png, .pdf and *_points.csv)
"$PY" -u scripts/plot_h3.py --out-dir figures/h3
"$PY" -u scripts/plot_h4.py --out-dir figures/h4 --joint-step "${JSTEP:-600}" --frozen-step "${FSTEP:-400}"
ls figures/h3 figures/h4
