#!/bin/bash -l
# Stage A' — order-sensitivity probe. One (model × engine × pool × variant × seed)
# cell. Reuses cached SERPs (no HTTP). Self-resubmits via afterany dependency
# on walltime/error, mirroring the main rerank chain.
#
# Required exports: MODEL, ENGINE, POOL, PROMPT_VARIANT, SEED
# Optional:         TOP_N (default 10), MAX_KEYWORDS (default unlimited),
#                   ATTEMPT, MAX_ATTEMPTS
#
# Submit (or use scripts/slurm/dispatch_all.sh --only order_probe):
#   sbatch --account=$JUWELS_ACCOUNT \
#     --export=ALL,MODEL=meta-llama/Llama-3.3-70B-Instruct,ENGINE=searxng,POOL=50,PROMPT_VARIANT=neutral,SEED=42 \
#     scripts/slurm/run_order_probe.sbatch

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=23:55:00
#SBATCH --job-name=geodml-orderprobe
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# shellcheck source=scripts/slurm/_common.sh
source "$SLURM_SUBMIT_DIR/scripts/slurm/_common.sh"

: "${MODEL:?MODEL must be set, e.g. meta-llama/Llama-3.3-70B-Instruct}"
: "${ENGINE:?ENGINE must be set: searxng or ddg}"
: "${POOL:?POOL must be set: 20 or 50}"
: "${PROMPT_VARIANT:?PROMPT_VARIANT must be set: biased or neutral}"
: "${SEED:?SEED must be set (e.g. 42 or 123)}"
: "${TOP_N:=10}"

MODEL_TAG="${MODEL##*/}"
RUN_ID="${ENGINE}_${MODEL_TAG}_serp${POOL}_top${TOP_N}_${PROMPT_VARIANT}"
OUTDIR="$GEODML_DATA_ROOT/data/order_probe"
mkdir -p "$OUTDIR"
OUT_JSONL="$OUTDIR/${RUN_ID}_seed${SEED}.jsonl"

echo "[order_probe] model=$MODEL engine=$ENGINE pool=$POOL variant=$PROMPT_VARIANT seed=$SEED"
echo "[order_probe] run_id=$RUN_ID"
echo "[order_probe] output=$OUT_JSONL"

set +e
PROMPT_VARIANT="$PROMPT_VARIANT" python -m interpretability.pipeline.order_probe \
  --engine "$ENGINE" \
  --pool "$POOL" \
  --top-n "$TOP_N" \
  --model "$MODEL" \
  --backend local \
  --variant "$PROMPT_VARIANT" \
  --seed "$SEED" \
  --resume \
  ${MAX_KEYWORDS:+--max-keywords "$MAX_KEYWORDS"}
RC=$?
set -e

echo "[order_probe] python rc=$RC"
if [ "$RC" -eq 0 ]; then
  touch "$OUTDIR/.done_${RUN_ID}_seed${SEED}"
  echo "[order_probe] complete -- chain stops here."
  exit 0
fi

chain_resubmit "$0" \
  MODEL="$MODEL" ENGINE="$ENGINE" POOL="$POOL" \
  PROMPT_VARIANT="$PROMPT_VARIANT" SEED="$SEED" TOP_N="$TOP_N" \
  ${MAX_KEYWORDS:+MAX_KEYWORDS="$MAX_KEYWORDS"} \
  || exit "$RC"
exit 0
