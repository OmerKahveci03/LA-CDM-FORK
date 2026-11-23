# One-time: a small plotting environment, separate from the training one (envs/la-cdm), so that
# installing matplotlib can never change the numpy/torch versions training depends on.
#   sbatch --export=ALL,SCRIPT=jobs/setup_plot_env.sh slurm/cpu_script.sbatch
# Creates /blue/prismap-ai-core/omerkahveci/envs/lacdm-plot (used by jobs/figures_h3_h4.sh).
PLOT_ENV=/blue/prismap-ai-core/omerkahveci/envs/lacdm-plot
if [[ ! -x "$PLOT_ENV/bin/python" ]]; then
  "$PY" -m venv "$PLOT_ENV"
fi
"$PLOT_ENV/bin/python" -m pip install --upgrade pip
"$PLOT_ENV/bin/python" -m pip install matplotlib numpy
"$PLOT_ENV/bin/python" -c "import matplotlib, numpy; print('matplotlib', matplotlib.__version__, 'numpy', numpy.__version__)"
