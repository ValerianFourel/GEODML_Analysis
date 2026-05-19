#!/bin/bash -l
# One probing chunk: one model, both frames in one job (probing is
# forward-only so it's cheap — both frames typically finish well within
# 24 h). The script writes a single probing_results.csv with a `frame`
# column so Figure C plots both curves on shared axes.
#
# Required exports: MODEL
# Optional:         SAMPLE_N (default huge), T7_KWS (default huge), FRAME
#                   (default both), RUN_FILTER, ATTEMPT, MAX_ATTEMPTS
#
# RUN_FILTER must match the model:
#   Llama-3.3-70B-Instruct -> searxng_Llama-3.3-70B-Instruct_serp50_top10
#   Qwen2.5-72B-Instruct   -> searxng_Qwen2.5-72B-Instruct_serp50_top10

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=05:55:00
#SBATCH --job-name=geodml-prob
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# JSC's submit filter blocks #SBATCH --requeue (see run_ablation.sbatch).

# shellcheck source=scripts/slurm/_common.sh
source "$SLURM_SUBMIT_DIR/scripts/slurm/_common.sh"

: "${MODEL:?MODEL must be set, e.g. meta-llama/Llama-3.3-70B-Instruct}"
: "${SAMPLE_N:=100000}"
: "${T7_KWS:=100000}"
: "${FRAME:=both}"
: "${PROMPT_VARIANT:=biased}"
export PROMPT_VARIANT

MODEL_TAG="${MODEL##*/}"
OUTDIR="$SLURM_SUBMIT_DIR/interpretability/output/probing_${MODEL_TAG}_${PROMPT_VARIANT}"
mkdir -p "$OUTDIR"

# Auto-pick the matching SearXNG run if RUN_FILTER wasn't passed. Pull from
# the variant-suffixed run dir so probing reads the right rerank output.
if [ -z "${RUN_FILTER:-}" ]; then
  RUN_FILTER="searxng_${MODEL_TAG}_serp50_top10_${PROMPT_VARIANT}"
fi

echo "[probing] model=$MODEL frame=$FRAME sample_n=$SAMPLE_N t7_kws=$T7_KWS variant=$PROMPT_VARIANT"
echo "[probing] run_filter=$RUN_FILTER"
echo "[probing] outdir=$OUTDIR"

set +e
python -m interpretability.probing \
  --model "$MODEL" \
  --sample-n "$SAMPLE_N" \
  --t7-keywords "$T7_KWS" \
  --frame "$FRAME" \
  --run-filter "$RUN_FILTER" \
  --resume \
  --output-dir "$OUTDIR"
RC=$?
set -e

echo "[probing] python rc=$RC"
if [ "$RC" -eq 0 ]; then
  touch "$OUTDIR/.done_${MODEL_TAG}"
  echo "[probing] complete -- chain stops here."
  exit 0
fi

chain_resubmit "$0" \
  MODEL="$MODEL" SAMPLE_N="$SAMPLE_N" T7_KWS="$T7_KWS" FRAME="$FRAME" \
  RUN_FILTER="$RUN_FILTER" PROMPT_VARIANT="$PROMPT_VARIANT" \
  || exit "$RC"
exit 0
