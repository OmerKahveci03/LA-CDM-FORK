#!/bin/bash
#SBATCH --job-name=lacdm-ext-v2
#SBATCH --output=logs/lacdm_ext_v2_%j.out
#SBATCH --error=logs/lacdm_ext_v2_%j.out
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00

# Edward Kemp's external validation (test set v2: 152 audited CC BY case-report patients, NOT MIMIC),
# run on our trained agents. Adapted from his slurm/ext_v2/trained_ext_v2.sbatch (see edward/README.md);
# the evaluation calls are unchanged. Only the test file and lab mapping differ from our own test runs:
#   dataset.test_data_file=data/ext_v2/ext_v2_test_strict_l3sum.csv
#   dataset.lab_test_mapping_file=data/ext_v2/ext_v2_lab_test_mapping.csv
#
# Arms (an arm runs only if all its files exist; existing results are never overwritten):
#   baseline  attempt 2 step-600 DA, its own HA                       src.evaluate
#   h3a       H3a DA step 700 + frozen separately fine-tuned HA        src.evaluate_frozen_ha
#   h4b       step-600 DA + CLM HA (fine-tuned backbone + stage-1 heads) src.contrastive_ha.evaluate
#   h4c       joint DA step 600 + the heads saved with it              src.contrastive_ha.evaluate
#
#   sbatch edward/trained_ext_v2.sbatch                         # all four arms (about 1 hour each)
#   sbatch --export=ALL,ARMS=baseline edward/trained_ext_v2.sbatch
#   sbatch --export=ALL,SMOKE=1 edward/trained_ext_v2.sbatch    # small sample, outputs <arm>_smoke
# Output: outputs/ext_v2_trained/<arm>/ (metrics.json, test_episodes.csv, test_summary.json, eval_output.log,
#         da_ha_disagreement.json once edward/da_ha_disagreement.py is present)
# Changes from Edward's version: runs from this repo by default (REPO, PY, FILES, DATA), log in logs/,
# 24G memory (our measured peak is under 8 GB), and the final chgrp covers only outputs/ext_v2_trained
# (his version also changed the group of the whole logs/ folder, which holds MIMIC-derived text).

set -euo pipefail
REPO="${REPO:-/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK}"
PY="${PY:-/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python}"
FILES="${FILES:-$REPO}"
DATA="${DATA:-$REPO/data/ext_v2}"
ARMS="${ARMS:-baseline h3a h4b h4c}"
TEST="$DATA/ext_v2_test_strict_l3sum.csv"
MAP="$DATA/ext_v2_lab_test_mapping.csv"
cd "$REPO"
mkdir -p logs
export VLLM_USE_V1=0 HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_HOME:-$REPO/.hf_cache} TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
[[ -f "$TEST" && -f "$MAP" ]] || { echo "Missing $TEST or $MAP (copy Edward's data; edward/README.md)" >&2; exit 2; }
COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src configs prompts || echo '-dirty')"
echo "[job] ${SLURM_JOB_ID:-local} on $(hostname), GPUs ${CUDA_VISIBLE_DEVICES:-none}, commit $COMMIT, $(date)"
echo "[job] files from $FILES; test $TEST; arms: $ARMS"

O=$FILES/outputs
BASE_DA=$O/cost_sweep/cost_1_attempt2/train/adapters/step_0600
H2=$O/ha_sft/seed_0/adapter
H3A_DA=$O/h3/h3a_ha0/train/adapters/step_0700
HEADS_H2=$O/contrastive/heads/h2/heads.pt
JOINT_DA=$O/contrastive/joint/joint_h2/train/adapters/step_0600
SMALL=false; SUFFIX=""
if [[ "${SMOKE:-0}" == "1" ]]; then SMALL=true; SUFFIX=_smoke; fi
COMMON=(dataset.test_data_file="$TEST" dataset.lab_test_mapping_file="$MAP" dataset.small_sample=$SMALL
        "reward_function.reward_functions.2.multiplier=1" evaluation.base_model=false
        training.vllm_gpu_memory_utilization=0.9 wandb.mode=disabled)

have () { for p in "$@"; do [[ -e "$p" ]] || { echo "[job] skip: missing $p"; return 1; }; done; }
run () {  # arm, module, extra hydra args...
  local arm=$1 mod=$2; shift 2
  local out="outputs/ext_v2_trained/$arm$SUFFIX"
  if [[ -f "$out/metrics.json" ]]; then echo "[job] $arm: $out/metrics.json exists, skipped"; return 0; fi
  mkdir -p "$out"
  echo "[job] $arm ($mod) -> $out ($(date))"
  "$PY" -m accelerate.commands.launch --num_processes 1 --num_machines 1 --mixed_precision no --dynamo_backend no \
      -m "$mod" hydra.run.dir="$out" "hydra.output_subdir=.hydra/$(date +%Y%m%d-%H%M%S)" \
      "experiment.name=la-cdm-ext-v2-$arm" evaluation.metrics_output_file="$out/metrics.json" "${COMMON[@]}" "$@" \
      > "$out/eval_output.log" 2>&1 || { echo "[job] $arm FAILED, see $out/eval_output.log"; return 0; }
  echo "[job] $arm finished ($(date))"
}

for arm in $ARMS; do
  case $arm in
    baseline) have "$BASE_DA/adapter_config.json" && run baseline src.evaluate evaluation.adapter_dir="$BASE_DA" ;;
    h3a) have "$H3A_DA/adapter_config.json" "$H2/adapter_config.json" &&
           run h3a src.evaluate_frozen_ha evaluation.adapter_dir="$H3A_DA" evaluation.frozen_ha_dir="$H2" ;;
    h4b) have "$BASE_DA/adapter_config.json" "$H2/adapter_config.json" "$HEADS_H2" &&
           run h4b src.contrastive_ha.evaluate evaluation.adapter_dir="$BASE_DA" +contrastive=defaults \
               contrastive.backbone=h2 contrastive.h2_adapter_dir="$H2" contrastive.heads_path="$HEADS_H2" ;;
    h4c) have "$JOINT_DA/adapter_config.json" "$JOINT_DA/contrastive_heads.pt" "$H2/adapter_config.json" &&
           run h4c src.contrastive_ha.evaluate evaluation.adapter_dir="$JOINT_DA" +contrastive=defaults \
               contrastive.backbone=h2 contrastive.h2_adapter_dir="$H2" \
               contrastive.heads_path="$JOINT_DA/contrastive_heads.pt" ;;
    *) echo "[job] unknown arm $arm" ;;
  esac
done

# DA-HA disagreement table for every finished arm (Edward's script; CPU, standard library). Only from this repo.
AN=$REPO/edward/da_ha_disagreement.py
if [[ -f "$AN" ]]; then
  for m in outputs/ext_v2_trained/*/metrics.json; do
    d=$(dirname "$m"); e=$(ls "$d"/*episodes*.csv 2>/dev/null | head -1 || true)
    [[ -n "$e" ]] && "$PY" "$AN" "$e" --name "$(basename "$d")" --out "$d/da_ha_disagreement.json" || true
  done
else
  echo "[job] no $AN yet; disagreement tables skipped"
fi
chgrp -R cai6734 outputs/ext_v2_trained 2>/dev/null || true   # results on Edward's data only, never logs/
echo "[job] done $(date)"
