# Figures for H3 and H4 (docs/HYPOTHESIS_H3.md, docs/HYPOTHESIS_H4.md).
#   sbatch --export=ALL,SCRIPT=jobs/figures_h3_h4.sh slurm/cpu_script.sbatch
#   optional: JSTEP / FSTEP = the joint / frozen arm's validation-chosen step (default 600 / 400)
#             MAX_STEP = plot H4 training curves only up to this step (e.g. 700); OUT = H4 folder
# Needs the plotting environment once: sbatch --export=ALL,SCRIPT=jobs/setup_plot_env.sh slurm/cpu_script.sbatch
# Output: figures/h3/ and figures/h4/ (or $OUT) (.png, .pdf and *_points.csv)
PLOT_PY=/blue/prismap-ai-core/omerkahveci/envs/lacdm-plot/bin/python
[[ -x "$PLOT_PY" ]] || { echo "No plotting environment; run jobs/setup_plot_env.sh first" >&2; exit 2; }
OUT="${OUT:-figures/h4}"
"$PLOT_PY" -u scripts/plot_h3.py --out-dir figures/h3
"$PLOT_PY" -u scripts/plot_h4.py --out-dir "$OUT" --joint-step "${JSTEP:-600}" --frozen-step "${FSTEP:-400}" \
  --max-step "${MAX_STEP:-0}"
ls figures/h3 "$OUT"
