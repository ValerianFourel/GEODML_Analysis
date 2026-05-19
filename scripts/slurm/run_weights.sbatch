#!/bin/bash -l
# Weight-analysis chunk: one model, all eligible keywords. Forward-only,
# typically 1.5-2.5 h, so a single 24 h walltime is plenty.
#
# Required exports: MODEL
# Optional:         SAMPLE_N (default huge), ATTEMPT, MAX_ATTEMPTS

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=05:55:00
#SBATCH --job-name=geodml-wgt
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# JSC's submit filter blocks #SBATCH --requeue (see run_ablation.sbatch).

# shellcheck source=scripts/slurm/_common.sh
source "$SLURM_SUBMIT_DIR/scripts/slurm/_common.sh"

: "${MODEL:?MODEL must be set, e.g. meta-llama/Llama-3.3-70B-Instruct}"
: "${SAMPLE_N:=100000}"
: "${PROMPT_VARIANT:=biased}"
export PROMPT_VARIANT

MODEL_TAG="${MODEL##*/}"
OUTDIR="$SLURM_SUBMIT_DIR/interpretability/output/weights_${MODEL_TAG}_${PROMPT_VARIANT}"
mkdir -p "$OUTDIR"

echo "[weights] model=$MODEL sample_n=$SAMPLE_N variant=$PROMPT_VARIANT"
echo "[weights] outdir=$OUTDIR"

set +e
python -m interpretability.weight_analysis \
  --model "$MODEL" \
  --sample-n "$SAMPLE_N" \
  --variant "$PROMPT_VARIANT" \
  --resume \
  --output-dir "$OUTDIR"
RC=$?
set -e

echo "[weights] python rc=$RC"
if [ "$RC" -eq 0 ]; then
  touch "$OUTDIR/.done_${MODEL_TAG}"
  echo "[weights] complete -- chain stops here."
  exit 0
fi

chain_resubmit "$0" \
  MODEL="$MODEL" SAMPLE_N="$SAMPLE_N" \
  PROMPT_VARIANT="$PROMPT_VARIANT" \
  || exit "$RC"
exit 0
