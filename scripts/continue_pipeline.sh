#!/usr/bin/env bash
# Run Stage B → C → D → figures on any CPU box (laptop or vast.ai), starting
# from the cluster snapshot already on disk under $GEODML_DATA_ROOT.
#
# Idempotent: re-runs skip cells whose output parquets already exist (and are
# non-empty). Safe to interrupt and resume — features.py / dml.py pass
# --resume.
#
# What it does NOT do:
#   - The 8 missing rerank cells (ddg × passage variants). Those need ddg
#     html_cache, which doesn't exist anywhere.
#   - The 16 missing order-probe cells (same blocker).
#   - The 6 missing probing cells. Probing extracts per-layer hidden states
#     from the 70B model — no API exposes those, GPU only.
#
# Required env:
#   GEODML_DATA_ROOT   absolute path to the unzipped data dir
#                      (must contain data/runs/*, data/serp/*, etc.)
#
# Optional env:
#   ENGINES            "searxng ddg"               (default)
#   POOLS              "20 50"                     (default)
#   VARIANTS           "biased neutral biased_passage neutral_passage"
#   FEATURES_DEVICE    "cpu" / "cuda" / "mps"      (default cpu)
#   FEATURES_MAX_KW    cap keywords per cell for smoke-testing (default off)
#   SKIP_FIGURES       "1" to skip make_figures
#   FORCE              "1" to ignore existing outputs and recompute

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${GEODML_DATA_ROOT:-}" ]; then
  echo "ERROR: GEODML_DATA_ROOT is not set. Export the absolute path to your data dir."
  echo "  e.g. export GEODML_DATA_ROOT=$HOME/geodml-restore/geodml_data"
  exit 2
fi
export GEODML_DATA_ROOT

: "${ENGINES:=searxng ddg}"
: "${POOLS:=20 50}"
: "${VARIANTS:=biased neutral biased_passage neutral_passage}"
: "${FEATURES_DEVICE:=cpu}"
: "${FEATURES_MAX_KW:=}"
: "${SKIP_FIGURES:=}"
: "${FORCE:=}"
export FEATURES_DEVICE

read -r -a ENGINES_ARR  <<<"$ENGINES"
read -r -a POOLS_ARR    <<<"$POOLS"
read -r -a VARIANTS_ARR <<<"$VARIANTS"

step()  { printf '\n\033[1m══════ %s ══════\033[0m\n' "$*"; }
ok()    { printf '  \033[32m✓\033[0m %s\n' "$*"; }
skip()  { printf '  \033[33m⊘\033[0m %s\n' "$*"; }
go()    { printf '  \033[36m→\033[0m %s\n' "$*"; }
fail()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }

# Treat as "already done" only if non-empty AND not forced.
done_already() {
  [ -z "$FORCE" ] && [ -s "$1" ]
}

has_html_cache() {
  local engine="$1" pool="$2"
  shopt -s nullglob
  for d in "$GEODML_DATA_ROOT/data/runs/${engine}_"*"_serp${pool}_top10"/phase2; do
    if [ -e "$d/html_cache.tar.gz" ] || [ -d "$d/html_cache" ]; then
      shopt -u nullglob
      return 0
    fi
  done
  shopt -u nullglob
  return 1
}

step "Pre-flight"
python -c "import pandas, sklearn, lightgbm, doubleml; print('  python deps OK')" \
  || { fail "missing core deps; pip install -r requirements.txt"; exit 2; }
echo "  GEODML_DATA_ROOT = $GEODML_DATA_ROOT"
[ -d "$GEODML_DATA_ROOT/data/serp" ] || { fail "no $GEODML_DATA_ROOT/data/serp — wrong path?"; exit 2; }
[ -d "$GEODML_DATA_ROOT/data/runs" ] || { fail "no $GEODML_DATA_ROOT/data/runs — wrong path?"; exit 2; }

step "Stage B — features (variant-agnostic)"
B_done=0; B_skip=0
for engine in "${ENGINES_ARR[@]}"; do
  for pool in "${POOLS_ARR[@]}"; do
    out="$GEODML_DATA_ROOT/data/features/features_${engine}_top${pool}.parquet"
    if done_already "$out"; then
      ok "${engine}/pool=${pool} already done ($(du -h "$out" | cut -f1))"
      B_done=$((B_done + 1))
      continue
    fi
    if ! has_html_cache "$engine" "$pool"; then
      skip "${engine}/pool=${pool}: no html_cache (expected for ddg; merge will skip)"
      B_skip=$((B_skip + 1))
      continue
    fi
    go "${engine}/pool=${pool}: extracting features (CPU; first run loads embedder)"
    cmd=(python -m interpretability.pipeline.features
         --engine "$engine" --pool "$pool"
         --device "$FEATURES_DEVICE" --resume)
    [ -n "$FEATURES_MAX_KW" ] && cmd+=(--max-keywords "$FEATURES_MAX_KW")
    if "${cmd[@]}"; then
      ok "${engine}/pool=${pool} done"
      B_done=$((B_done + 1))
    else
      fail "${engine}/pool=${pool} failed (rc=$?). Continuing — merge will skip."
    fi
  done
done
echo "  Stage B summary: ${B_done} done, ${B_skip} skipped (ddg/no-html)"

step "Stage C — merge per variant"
C_done=0
for variant in "${VARIANTS_ARR[@]}"; do
  out="$GEODML_DATA_ROOT/data/main/full_experiment_data_${variant}.parquet"
  if done_already "$out"; then
    ok "${variant} already done ($(du -h "$out" | cut -f1))"
    C_done=$((C_done + 1))
    continue
  fi
  go "merging variant=${variant}"
  if python scripts/build_main_table.py --variant "$variant"; then
    ok "${variant} done"
    C_done=$((C_done + 1))
  else
    fail "${variant} merge failed (rc=$?). Likely no features parquet for any cell of this variant."
  fi
done
echo "  Stage C summary: ${C_done}/4 variants merged"

step "Stage D — DML per variant"
D_done=0
for variant in "${VARIANTS_ARR[@]}"; do
  in_path="$GEODML_DATA_ROOT/data/main/full_experiment_data_${variant}.parquet"
  out_path="$GEODML_DATA_ROOT/data/dml_results/dml_results_long_${variant}.parquet"
  if ! [ -s "$in_path" ]; then
    skip "${variant}: no main table from Stage C (likely missing features)"
    continue
  fi
  if done_already "$out_path"; then
    ok "${variant} already done ($(du -h "$out_path" | cut -f1))"
    D_done=$((D_done + 1))
    continue
  fi
  go "DML variant=${variant} (resume on; LightGBM + RF, ~5–15 min/variant)"
  if python -m interpretability.pipeline.dml --variant "$variant" --resume; then
    ok "${variant} done"
    D_done=$((D_done + 1))
  else
    fail "${variant} DML failed (rc=$?)."
  fi
done
echo "  Stage D summary: ${D_done}/4 variants fitted"

if [ -z "$SKIP_FIGURES" ]; then
  step "Figures"
  if python -m interpretability.make_figures; then
    ok "figures regenerated under interpretability/output/plots/"
  else
    skip "make_figures failed (non-fatal — re-run after all variants merge)"
  fi
fi

step "Audit (after)"
python scripts/audit_pipeline.py
