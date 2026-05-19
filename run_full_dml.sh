#!/bin/bash
# End-to-end DML analysis after Stage A (rerank) + A' (order_probe) are done.
#
# Sequence:
#   1. backfill_precision     -- normalize llm_parameters.precision on all records
#   2. audit_status (pre)     -- snapshot cell-completeness before pipeline
#   3. continue_pipeline.sh   -- Stage B (features) → C (merge per variant) →
#                                D (DML per variant) → figures → archive
#   4. make_results_summary   -- compact results table for the paper
#   5. audit_status (post)    -- final snapshot
#
# All steps are idempotent. Re-runs skip cells whose parquets already exist
# (set FORCE=1 to override). Safe to interrupt and re-run.
#
# Usage:
#   ./run_full_dml.sh                          # run everything
#   FEATURES_DEVICE=cuda ./run_full_dml.sh     # GPU for the sentence-transformer
#                                                embedder in Stage B (much faster)
#   SKIP_BACKFILL=1 ./run_full_dml.sh          # skip step 1
#   SKIP_SUMMARY=1  ./run_full_dml.sh          # skip step 4
#   FORCE=1 ./run_full_dml.sh                  # ignore existing outputs (full redo)
#
# Where to run on JUPITER:
#   The pipeline is CPU-bound (LightGBM + DoubleML cross-fitting). Do NOT run
#   it on the login node — it'll get killed. Use salloc or sbatch:
#
#     salloc --nodes=1 --time=04:00:00 --account=scifi -p booster --gres=gpu:1
#     srun --pty bash -l
#     cd /e/project1/scifi/$USER/GEODML_Analysis
#     jutil env activate -p scifi
#     module load Stages/2026 GCC Python CUDA
#     source .venv/bin/activate
#     set -a; source .env; set +a
#     FEATURES_DEVICE=cuda ./run_full_dml.sh

set -uo pipefail
cd "$(dirname "$0")"

[ -f .env ] && { set -a; source .env; set +a; }
: "${GEODML_DATA_ROOT:?GEODML_DATA_ROOT must be set (export it or put it in .env)}"

SKIP_BACKFILL="${SKIP_BACKFILL:-0}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"

step() { printf '\n\033[1m═════ %s ═════\033[0m\n' "$*"; }

if [ "$SKIP_BACKFILL" != "1" ]; then
  step "1/5  backfill precision metadata"
  python scripts/backfill_precision.py --root "$GEODML_DATA_ROOT" --include-recent
else
  step "1/5  backfill (skipped via SKIP_BACKFILL=1)"
fi

step "2/5  audit BEFORE pipeline"
python scripts/audit_status.py | tail -40 || true

step "3/5  Stage B → C → D (continue_pipeline.sh)"
bash scripts/continue_pipeline.sh

if [ "$SKIP_SUMMARY" != "1" ]; then
  step "4/5  results summary"
  python scripts/make_results_summary.py --data-root "$GEODML_DATA_ROOT" || true
else
  step "4/5  results summary (skipped via SKIP_SUMMARY=1)"
fi

step "5/5  audit AFTER pipeline"
python scripts/audit_status.py | tail -40 || true

step "DONE"
echo "Outputs:"
echo "  features:     $GEODML_DATA_ROOT/data/features/"
echo "  main tables:  $GEODML_DATA_ROOT/data/main/"
echo "  DML results:  $GEODML_DATA_ROOT/data/dml_results/"
echo "  figures:      interpretability/output/plots/"
echo "  archive:      $(ls -1t archives/local_results_*.zip 2>/dev/null | head -1 || echo '(none yet)')"
echo
echo "Next: upload the archive to HF (see the upload command printed by continue_pipeline.sh)."
