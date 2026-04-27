#!/bin/bash -l
# One saliency chunk: one model, one frame (full or robust_winners), all
# eligible keywords. We split per-frame here because saliency does a backward
# pass through a 70B+ model, so each frame can take 8-12 h on its own —
# running both in one job risks blowing the 24 h walltime cap.
#
# Required exports: MODEL, FRAME (full | robust_winners)
# Optional:         SAMPLE_N (default huge so all eligible kws are taken),
#                   ATTEMPT, MAX_ATTEMPTS

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=23:55:00
#SBATCH --job-name=geodml-sal
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# JSC's submit filter blocks #SBATCH --requeue (see run_ablation.sbatch).

# shellcheck source=scripts/slurm/_common.sh
source "$SLURM_SUBMIT_DIR/scripts/slurm/_common.sh"

: "${MODEL:?MODEL must be set, e.g. meta-llama/Llama-3.3-70B-Instruct}"
: "${FRAME:?FRAME must be set: full | robust_winners}"
: "${SAMPLE_N:=100000}"

MODEL_TAG="${MODEL##*/}"
OUTDIR="$SLURM_SUBMIT_DIR/interpretability/output/saliency_${MODEL_TAG}"
mkdir -p "$OUTDIR"

echo "[saliency] model=$MODEL frame=$FRAME sample_n=$SAMPLE_N"
echo "[saliency] outdir=$OUTDIR"

set +e
python -m interpretability.saliency \
  --model "$MODEL" \
  --sample-n "$SAMPLE_N" \
  --frame "$FRAME" \
  --grad-checkpoint \
  --max-len 2048 \
  --resume \
  --output-dir "$OUTDIR"
RC=$?
set -e

echo "[saliency] python rc=$RC"
if [ "$RC" -eq 0 ]; then
  touch "$OUTDIR/.done_${MODEL_TAG}_${FRAME}"
  echo "[saliency] complete -- chain stops here."
  exit 0
fi

chain_resubmit "$0" \
  MODEL="$MODEL" FRAME="$FRAME" SAMPLE_N="$SAMPLE_N" \
  || exit "$RC"
exit 0
