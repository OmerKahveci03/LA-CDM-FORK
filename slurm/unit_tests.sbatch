#!/bin/bash
#SBATCH --job-name=lacdm-tests
#SBATCH --output=logs/lacdm_tests_%j.out
#SBATCH --error=logs/lacdm_tests_%j.out
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Run unit tests on a compute node (CPU only; the tests use tiny models).
#
#   sbatch slurm/unit_tests.sbatch                                   # the whole suite
#   sbatch --export=ALL,TESTS="tests.test_contrastive_ha tests.test_frozen_ha" slurm/unit_tests.sbatch
#
# Output: logs/lacdm_tests_<jobid>.out (one file for stdout and stderr). Ends with OK or FAILED.

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
cd "$REPO_DIR"
mkdir -p logs

# Match the threads to the allocated cores (otherwise torch oversubscribes the CPU).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUDA_VISIBLE_DEVICES=""

echo "[SLURM] Job: ${SLURM_JOB_ID:-local}  Commit: $(git rev-parse --short HEAD)  Host: $(hostname)  Started: $(date)"
if [[ -n "${TESTS:-}" ]]; then
    # shellcheck disable=SC2086  # TESTS is a space-separated list of test modules
    "$PYTHON_BIN" -m unittest -v $TESTS
else
    "$PYTHON_BIN" -m unittest discover -v -s tests
fi
echo "[SLURM] Finished: $(date)"
