#!/bin/bash -l
# One ablation chunk: one model, one treatment, both frames, all keywords.
# Self-resubmits via afterany dependency on walltime/error until done.
#
# Required exports: MODEL, TREATMENT
# Optional:         SAMPLE_N (default huge so all eligible kws are taken),
#                   FRAME (default both), ATTEMPT, MAX_ATTEMPTS
#
# Submit (or use scripts/slurm/dispatch_all.sh):
#   sbatch --account=$JUWELS_ACCOUNT \
#          --export=ALL,MODEL=meta-llama/Llama-3.3-70B-Instruct,TREATMENT=T7_source_earned \
#          scripts/slurm/run_ablation.sbatch

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=23:55:00
#SBATCH --job-name=geodml-abl
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Note: JSC's submit filter blocks #SBATCH --requeue. Node-failure recovery
# isn't automatic — re-run dispatch_all.sh if a node dies. Walltime / transient
# errors are still handled by the chain_resubmit chain at the bottom.

# shellcheck source=scripts/slurm/_common.sh
source "$SLURM_SUBMIT_DIR/scripts/slurm/_common.sh"

: "${MODEL:?MODEL must be set, e.g. meta-llama/Llama-3.3-70B-Instruct}"
: "${TREATMENT:?TREATMENT must be set, e.g. T7_source_earned}"
: "${SAMPLE_N:=100000}"
: "${FRAME:=both}"
: "${PROMPT_VARIANT:=biased}"
export PROMPT_VARIANT

MODEL_TAG="${MODEL##*/}"
OUTDIR="$SLURM_SUBMIT_DIR/interpretability/output/ablation_${TREATMENT}_${MODEL_TAG}_${PROMPT_VARIANT}"
mkdir -p "$OUTDIR"

echo "[ablation] model=$MODEL treatment=$TREATMENT frame=$FRAME sample_n=$SAMPLE_N variant=$PROMPT_VARIANT"
echo "[ablation] outdir=$OUTDIR"

set +e
python -m interpretability.ablation \
  --backend local \
  --models "$MODEL" \
  --treatments "$TREATMENT" \
  --sample-n "$SAMPLE_N" \
  --frame "$FRAME" \
  --resume \
  --output-dir "$OUTDIR"
RC=$?
set -e

echo "[ablation] python rc=$RC"
if [ "$RC" -eq 0 ]; then
  touch "$OUTDIR/.done_${MODEL_TAG}_${TREATMENT}"
  echo "[ablation] complete -- chain stops here."
  exit 0
fi

chain_resubmit "$0" \
  MODEL="$MODEL" TREATMENT="$TREATMENT" SAMPLE_N="$SAMPLE_N" FRAME="$FRAME" \
  PROMPT_VARIANT="$PROMPT_VARIANT" \
  || exit "$RC"
exit 0
