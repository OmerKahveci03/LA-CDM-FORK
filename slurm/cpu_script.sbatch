#!/bin/bash
#SBATCH --job-name=lacdm-cpu
#SBATCH --output=logs/lacdm_cpu_%j.out
#SBATCH --error=logs/lacdm_cpu_%j.out
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Run a CPU-only shell script on a compute node (statistics, plots, small analyses), so that
# nothing heavier than cat/tail/sbatch runs on a login node.
#
#   sbatch --export=ALL,SCRIPT=jobs/h4a_stats.sh slurm/cpu_script.sbatch
#
# The script runs from the repository root with $PY set to the project's Python.
# Output: logs/lacdm_cpu_<jobid>.out

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
export PY=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
cd "$REPO_DIR"
mkdir -p logs
SCRIPT="${SCRIPT:?Set SCRIPT to a shell script (path relative to the repository root)}"
[[ -f "$SCRIPT" ]] || { echo "No script at $SCRIPT" >&2; exit 2; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUDA_VISIBLE_DEVICES=""
echo "[SLURM] Job: ${SLURM_JOB_ID:-local}  Commit: $(git rev-parse --short HEAD)  Script: $SCRIPT  Started: $(date)"
bash -euo pipefail "$SCRIPT"
echo "[SLURM] Finished: $(date)"
