#!/bin/bash -l
# Stage A — LLM rerank one (model × engine × pool × variant) cell of the grid.
# Uses cached SERPs from data/serp/phase0_*.parquet (no HTTP).
# Self-resubmits via afterany dependency on walltime/error.
#
# Required exports: MODEL, ENGINE, POOL, PROMPT_VARIANT
# Optional:         TOP_N (default 10), MAX_KEYWORDS (default unlimited),
#                   ATTEMPT, MAX_ATTEMPTS,
#                   LOCAL_PRECISION (default "full" = bf16; alt "4bit" = nf4)
#
# 70B at bf16 ≈ 140 GB; needs all 4 GPUs (already --gres=gpu:4 below).
# 70B at 4-bit nf4 ≈ 35 GB; can fit on 2 GPUs but slurm slot is still 4.
#
# Submit (or use scripts/slurm/dispatch_all.sh --only rerank):
#   sbatch --account=$JUWELS_ACCOUNT \
#     --export=ALL,MODEL=meta-llama/Llama-3.3-70B-Instruct,ENGINE=searxng,POOL=50,PROMPT_VARIANT=neutral,LOCAL_PRECISION=full \
#     scripts/slurm/run_rerank.sbatch

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=23:55:00
#SBATCH --job-name=geodml-rerank
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# shellcheck source=scripts/slurm/_common.sh
source "$SLURM_SUBMIT_DIR/scripts/slurm/_common.sh"

: "${MODEL:?MODEL must be set, e.g. meta-llama/Llama-3.3-70B-Instruct}"
: "${ENGINE:?ENGINE must be set: searxng or ddg}"
: "${POOL:?POOL must be set: 20 or 50}"
: "${PROMPT_VARIANT:?PROMPT_VARIANT must be set: biased or neutral}"
: "${TOP_N:=10}"

MODEL_TAG="${MODEL##*/}"
RUN_ID="${ENGINE}_${MODEL_TAG}_serp${POOL}_top${TOP_N}_${PROMPT_VARIANT}"
OUTDIR="$GEODML_DATA_ROOT/data/runs/$RUN_ID/phase2"
mkdir -p "$OUTDIR"

echo "[rerank] model=$MODEL engine=$ENGINE pool=$POOL variant=$PROMPT_VARIANT precision=${LOCAL_PRECISION:-full}"
echo "[rerank] run_id=$RUN_ID"
echo "[rerank] outdir=$OUTDIR"

# Pre-flight: skip if this cell is already at MAX_KEYWORDS (saves model-load time).
skip_if_at_max "$OUTDIR/keywords.jsonl" "${MAX_KEYWORDS:-0}" \
  "$OUTDIR/.done_rerank_${MODEL_TAG}_${PROMPT_VARIANT}"

set +e
PROMPT_VARIANT="$PROMPT_VARIANT" python -m interpretability.pipeline.rerank \
  --engine "$ENGINE" \
  --pool "$POOL" \
  --top-n "$TOP_N" \
  --model "$MODEL" \
  --backend local \
  --precision "${LOCAL_PRECISION:-full}" \
  --variant "$PROMPT_VARIANT" \
  --resume \
  ${MAX_KEYWORDS:+--max-keywords "$MAX_KEYWORDS"}
RC=$?
set -e

echo "[rerank] python rc=$RC"
if [ "$RC" -eq 0 ]; then
  touch "$OUTDIR/.done_rerank_${MODEL_TAG}_${PROMPT_VARIANT}"
  echo "[rerank] complete -- chain stops here."
  exit 0
fi

chain_resubmit "$0" \
  MODEL="$MODEL" ENGINE="$ENGINE" POOL="$POOL" \
  PROMPT_VARIANT="$PROMPT_VARIANT" TOP_N="$TOP_N" \
  LOCAL_PRECISION="${LOCAL_PRECISION:-full}" \
  ${MAX_KEYWORDS:+MAX_KEYWORDS="$MAX_KEYWORDS"} \
  || exit "$RC"
exit 0
