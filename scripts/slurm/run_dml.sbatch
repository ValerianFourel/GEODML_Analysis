#!/bin/bash -l
# Stage D — DoubleML PLR/IRM grid for one variant. CPU-only (LightGBM/RF
# nuisance learners; no GPU). Walltime ~50 min for the default grid.
#
# Required exports: PROMPT_VARIANT (biased or neutral)
# Optional:         DML_MEASUREMENT (default 'new'), DML_OUTCOMES (default rank_delta+post_rank),
#                   DML_METHODS (default plr), DML_LEARNERS (default lgbm rf),
#                   DML_SUBSETS (default POOLED+by_engine+by_model+by_pool),
#                   N_FOLDS (default 5), DML_OLS (set to 1 to also emit OLS rows),
#                   ATTEMPT, MAX_ATTEMPTS
#
# Submit:
#   sbatch --account=$JUWELS_ACCOUNT \
#     --export=ALL,PROMPT_VARIANT=neutral \
#     scripts/slurm/run_dml.sbatch

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=05:55:00
#SBATCH --job-name=geodml-dml
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# shellcheck source=scripts/slurm/_common.sh
source "$SLURM_SUBMIT_DIR/scripts/slurm/_common.sh"

: "${PROMPT_VARIANT:?PROMPT_VARIANT must be set: biased or neutral}"
: "${DML_MEASUREMENT:=new}"
: "${N_FOLDS:=5}"

# Convert space-separated strings into shell arrays for argparse nargs='*'.
read -r -a OUTCOMES_ARR  <<< "${DML_OUTCOMES:-rank_delta post_rank}"
read -r -a METHODS_ARR   <<< "${DML_METHODS:-plr}"
read -r -a LEARNERS_ARR  <<< "${DML_LEARNERS:-lgbm rf}"
read -r -a SUBSETS_ARR   <<< "${DML_SUBSETS:-POOLED by_engine by_model by_pool}"

OUTDIR="$GEODML_DATA_ROOT/data/dml_results"
mkdir -p "$OUTDIR"
mkdir -p "$GEODML_DATA_ROOT/data/main"

echo "[dml] variant=$PROMPT_VARIANT measurement=$DML_MEASUREMENT n_folds=$N_FOLDS"
echo "[dml] outcomes=${OUTCOMES_ARR[*]} methods=${METHODS_ARR[*]} learners=${LEARNERS_ARR[*]}"
echo "[dml] subsets=${SUBSETS_ARR[*]}"

# ── Stage C (merge) — pure pandas, ~30s, runs inline before DML.
# Only re-merge if the main table is missing or empty (cheap so default-on).
MAIN_PARQUET="$GEODML_DATA_ROOT/data/main/full_experiment_data_${PROMPT_VARIANT}.parquet"
if [ ! -s "$MAIN_PARQUET" ]; then
  echo "[dml] running Stage C (merge) -> $MAIN_PARQUET"
  python scripts/build_main_table.py --variant "$PROMPT_VARIANT" \
    ${EXTERNAL_FEATURES_PARQUET:+--external-features-parquet "$EXTERNAL_FEATURES_PARQUET"}
fi

set +e
python -m interpretability.pipeline.dml \
  --variant "$PROMPT_VARIANT" \
  --measurement "$DML_MEASUREMENT" \
  --outcomes "${OUTCOMES_ARR[@]}" \
  --methods  "${METHODS_ARR[@]}" \
  --learners "${LEARNERS_ARR[@]}" \
  --subsets  "${SUBSETS_ARR[@]}" \
  --n-folds "$N_FOLDS" \
  ${DML_OLS:+--ols} \
  --resume
RC=$?
set -e

echo "[dml] python rc=$RC"
if [ "$RC" -eq 0 ]; then
  touch "$OUTDIR/.done_dml_${PROMPT_VARIANT}"
  echo "[dml] complete -- chain stops here."
  exit 0
fi

chain_resubmit "$0" \
  PROMPT_VARIANT="$PROMPT_VARIANT" DML_MEASUREMENT="$DML_MEASUREMENT" N_FOLDS="$N_FOLDS" \
  ${DML_OUTCOMES:+DML_OUTCOMES="$DML_OUTCOMES"} \
  ${DML_METHODS:+DML_METHODS="$DML_METHODS"} \
  ${DML_LEARNERS:+DML_LEARNERS="$DML_LEARNERS"} \
  ${DML_SUBSETS:+DML_SUBSETS="$DML_SUBSETS"} \
  ${DML_OLS:+DML_OLS=1} \
  || exit "$RC"
exit 0
