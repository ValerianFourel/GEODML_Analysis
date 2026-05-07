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

# Lock in the same LLM config the cluster used. The local pipeline is B/C/D
# only (deterministic), but if any rerank-style step is re-invoked later,
# config.py is the single source of truth — and these values must match the
# cluster runs (temperature=0.1, max_tokens=500) for results to be comparable.
LLM_CFG=$(python -c "
from interpretability.pipeline import config as C
print(f'{C.LLM_TEMPERATURE}|{C.LLM_MAX_TOKENS}')
")
LLM_TEMP=${LLM_CFG%%|*}
LLM_TOKS=${LLM_CFG##*|}
echo "  LLM config       = temperature=${LLM_TEMP}, max_tokens=${LLM_TOKS} (matches cluster)"
if [ "$LLM_TEMP" != "0.1" ] || [ "$LLM_TOKS" != "500" ]; then
  fail "config.py drifted from cluster values (expected 0.1 / 500). Aborting."
  exit 2
fi
echo "  Stages B/C/D are deterministic (no LLM call). Temperature is preserved"
echo "  for any future rerank re-run via config.py."

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
python scripts/audit_pipeline.py | tee /tmp/audit_after.txt

step "Package results for upload"
DATE=$(date +%Y%m%d-%H%M)
ARCHIVES_DIR="$REPO_ROOT/archives"
mkdir -p "$ARCHIVES_DIR"
RESULT_ZIP="$ARCHIVES_DIR/local_results_${DATE}.zip"

# Stage only what this run produced — feature parquets, main tables, DML
# results, and freshly regenerated figures. NOT the full geodml_data dump
# (that was already uploaded as geodml_data_.zip).
TMP_STAGE=$(mktemp -d)
trap 'rm -rf "$TMP_STAGE"' EXIT

mkdir -p "$TMP_STAGE/geodml_data/data"
for sub in features main dml_results; do
  src="$GEODML_DATA_ROOT/data/$sub"
  if [ -d "$src" ]; then
    cp -R "$src" "$TMP_STAGE/geodml_data/data/"
  fi
done

if [ -d "$REPO_ROOT/interpretability/output/plots" ]; then
  mkdir -p "$TMP_STAGE/interpretability/output"
  cp -R "$REPO_ROOT/interpretability/output/plots" \
        "$TMP_STAGE/interpretability/output/"
fi

cat > "$TMP_STAGE/PROVENANCE.txt" <<PROV
GEODML local Stage B/C/D results bundle
========================================
generated:        $(date -u +%Y-%m-%dT%H:%M:%SZ)
host:             $(hostname)
user:             ${USER:-unknown}
git:              $(git -C "$REPO_ROOT" log -1 --pretty=format:'%h %s' 2>/dev/null || echo 'n/a')

Pipeline config (matches cluster runs):
  LLM_TEMPERATURE = ${LLM_TEMP}
  LLM_MAX_TOKENS  = ${LLM_TOKS}
  source:           interpretability/pipeline/config.py

Contents:
  geodml_data/data/features/    - Stage B parquets (variant-agnostic)
  geodml_data/data/main/        - Stage C merged main tables (per variant)
  geodml_data/data/dml_results/ - Stage D DML long tables (per variant)
  interpretability/output/plots/- regenerated paper figures

NOT included (already on the Hub):
  geodml_data/data/runs/*       - cluster rerank/order-probe outputs
  geodml_data/data/serp/*       - SERP inputs
  interpretability/output/      - Stage F per-cell CSVs (ablation, saliency,
                                  weights, probing 2/8) — already uploaded
                                  in interpretability_.zip

Stage F next steps (separate, GPU-only on JUWELS):
  6 missing probing cells (neutral, biased_passage, neutral_passage × Llama
  + Qwen). Probing extracts per-layer hidden states from the 70B model;
  forward-only but cannot run via inference API. Submit on JUWELS:
    sbatch --account=\$JUWELS_ACCOUNT --export=ALL,MODEL=meta-llama/Llama-3.3-70B-Instruct,PROMPT_VARIANT=neutral scripts/slurm/run_probing.sbatch
  ...repeated for the 6 missing (model × variant) combos.
PROV

( cd "$TMP_STAGE" && zip -qr "$RESULT_ZIP" . )
ok "wrote $(du -h "$RESULT_ZIP" | cut -f1) -> $RESULT_ZIP"

step "Upload command (review before running)"
cat <<UPLOAD
  # Push this run's results to the HF dataset (requires WRITE-scoped token):
  huggingface-cli login    # one-time, paste a write token
  hf upload ValerianFourel/geodml-papersize \\
    "$RESULT_ZIP" \\
    "archives/local_results_${DATE}.zip" \\
    --repo-type dataset \\
    --commit-message "local Stage B/C/D ${DATE}"
UPLOAD

step "What's left"
cat <<NEXT
  - Stage F probing: 6 cells × ≤24 h on 4×A100-40G each. JUWELS only.
    See PROVENANCE.txt inside $RESULT_ZIP for the sbatch invocation.
  - All other stages: complete in this snapshot.
NEXT
