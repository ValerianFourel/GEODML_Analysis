#!/bin/bash -l
# Stage B — deterministic features for one (engine × pool) cell.
# Variant-agnostic (page features don't depend on the prompt), so this
# emits to data/features/features_{engine}_top{pool}.parquet, shared across
# variants.
#
# Required exports: ENGINE, POOL
# Optional:         FEATURES_DEVICE (cpu/cuda; default auto), HTML_RUN_ID,
#                   ATTEMPT, MAX_ATTEMPTS
#
# Submit:
#   sbatch --account=$JUWELS_ACCOUNT \
#     --export=ALL,ENGINE=searxng,POOL=50 \
#     scripts/slurm/run_features.sbatch

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=11:55:00
#SBATCH --job-name=geodml-feat
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# shellcheck source=scripts/slurm/_common.sh
source "$SLURM_SUBMIT_DIR/scripts/slurm/_common.sh"

: "${ENGINE:?ENGINE must be set: searxng or ddg}"
: "${POOL:?POOL must be set: 20 or 50}"
: "${FEATURES_DEVICE:=cuda}"

OUTDIR="$GEODML_DATA_ROOT/data/features"
mkdir -p "$OUTDIR"

echo "[features] engine=$ENGINE pool=$POOL device=$FEATURES_DEVICE"
echo "[features] output dir=$OUTDIR"

set +e
FEATURES_DEVICE="$FEATURES_DEVICE" python -m interpretability.pipeline.features \
  --engine "$ENGINE" \
  --pool "$POOL" \
  --device "$FEATURES_DEVICE" \
  --resume \
  ${HTML_RUN_ID:+--html-run-id "$HTML_RUN_ID"}
RC=$?
set -e

echo "[features] python rc=$RC"
if [ "$RC" -eq 0 ]; then
  touch "$OUTDIR/.done_features_${ENGINE}_top${POOL}"
  echo "[features] complete -- chain stops here."
  exit 0
fi

chain_resubmit "$0" \
  ENGINE="$ENGINE" POOL="$POOL" FEATURES_DEVICE="$FEATURES_DEVICE" \
  ${HTML_RUN_ID:+HTML_RUN_ID="$HTML_RUN_ID"} \
  || exit "$RC"
exit 0
