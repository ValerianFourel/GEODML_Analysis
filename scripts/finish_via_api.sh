#!/usr/bin/env bash
# Finish everything that's API-recoverable from a CPU box.
# Designed for the case where you've lost cluster GPU access and need to
# show the maximum possible coverage to a stakeholder.
#
# What this fills in (no GPU needed):
#   - 4 rerank cells: ddg × pool=50 × {biased_passage, neutral_passage} × {Llama, Qwen}
#     using the legacy duckduckgo_*_serp50_top10 html_cache as the passage source
#   - 8 order-probe cells: same 4 cells × 2 seeds (42, 123)
#   - Stage A' summary parquet (re-aggregates after order_probe completes)
#   - Stage B (features) for searxng cells
#   - Stage C (merge) for all 4 variants
#   - Stage D (DML) for all 4 variants
#   - Paper figures
#
# What stays unfilled — needs GPU access you'll regain later:
#   - 4 rerank cells: ddg × pool=20 × passage variants (no ddg pool=20 HTML
#     exists anywhere; not API-recoverable either)
#   - 8 order-probe cells (same blocker)
#   - 6 probing cells (Stage F) — needs raw model hidden states
#
# Required env (set before running):
#   GEODML_DATA_ROOT  abs path to the unzipped data dir
#   OPENAI_API_KEY    DeepInfra/Together/Fireworks key
#   OPENAI_BASE_URL   provider base URL (see PROVIDER_PRESETS below)
#
# Optional env:
#   PROVIDER          "deepinfra" / "together" / "fireworks" — sets BASE_URL
#                     and model name overrides for you
#   SKIP_RERANK       "1" to skip the 4 ddg-passage rerank cells
#   SKIP_ORDER_PROBE  "1" to skip the 8 ddg-passage order-probe cells
#   SKIP_BCD          "1" to skip Stages B/C/D
#   FEATURES_DEVICE   "cpu" (default) / "mps" / "cuda"
#   MAX_KW            cap keywords per cell (smoke testing); empty = no cap

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

step() { printf '\n\033[1m══════ %s ══════\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
go()   { printf '  \033[36m→\033[0m %s\n' "$*"; }
skip() { printf '  \033[33m⊘\033[0m %s\n' "$*"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$*"; }

# ── Sanity checks ────────────────────────────────────────────────────────────
[ -n "${GEODML_DATA_ROOT:-}" ] || { fail "GEODML_DATA_ROOT not set"; exit 2; }
export GEODML_DATA_ROOT

# Provider presets — flip OPENAI_BASE_URL + model id depending on PROVIDER.
case "${PROVIDER:-}" in
  deepinfra)
    : "${OPENAI_BASE_URL:=https://api.deepinfra.com/v1/openai}"
    LLAMA_MODEL_API="meta-llama/Meta-Llama-3.1-70B-Instruct"  # closest hosted; check current catalog
    QWEN_MODEL_API="Qwen/Qwen2.5-72B-Instruct"
    ;;
  together)
    : "${OPENAI_BASE_URL:=https://api.together.xyz/v1}"
    LLAMA_MODEL_API="meta-llama/Llama-3.3-70B-Instruct-Turbo"
    QWEN_MODEL_API="Qwen/Qwen2.5-72B-Instruct-Turbo"
    ;;
  fireworks)
    : "${OPENAI_BASE_URL:=https://api.fireworks.ai/inference/v1}"
    LLAMA_MODEL_API="accounts/fireworks/models/llama-v3p3-70b-instruct"
    QWEN_MODEL_API="accounts/fireworks/models/qwen2p5-72b-instruct"
    ;;
  *)
    LLAMA_MODEL_API="${LLAMA_MODEL_API:-meta-llama/Llama-3.3-70B-Instruct}"
    QWEN_MODEL_API="${QWEN_MODEL_API:-Qwen/Qwen2.5-72B-Instruct}"
    ;;
esac

# We pass the canonical HF model id to the python (so run dirs match the
# cluster's), but rely on OPENAI_MODEL_OVERRIDE to map to provider id.
LLAMA_MODEL_HF="meta-llama/Llama-3.3-70B-Instruct"
QWEN_MODEL_HF="Qwen/Qwen2.5-72B-Instruct"

if [ -z "${SKIP_RERANK:-}" ] || [ -z "${SKIP_ORDER_PROBE:-}" ]; then
  [ -n "${OPENAI_API_KEY:-}" ] || { fail "OPENAI_API_KEY not set"; exit 2; }
  [ -n "${OPENAI_BASE_URL:-}" ] || { fail "OPENAI_BASE_URL not set (or set PROVIDER=...)"; exit 2; }
  export OPENAI_API_KEY OPENAI_BASE_URL
fi

step "Pre-flight"
echo "  GEODML_DATA_ROOT  = $GEODML_DATA_ROOT"
echo "  OPENAI_BASE_URL   = ${OPENAI_BASE_URL:-(unset; rerank/probe disabled)}"
echo "  Llama (HF id)     = $LLAMA_MODEL_HF"
echo "  Llama (API id)    = $LLAMA_MODEL_API"
echo "  Qwen  (HF id)     = $QWEN_MODEL_HF"
echo "  Qwen  (API id)    = $QWEN_MODEL_API"

python -c "
from interpretability.pipeline import config as C
assert C.LLM_TEMPERATURE == 0.1, C.LLM_TEMPERATURE
assert C.LLM_MAX_TOKENS == 500, C.LLM_MAX_TOKENS
print(f'  LLM config        = temperature={C.LLM_TEMPERATURE}, max_tokens={C.LLM_MAX_TOKENS} (matches cluster)')
" || { fail "config.py drifted from cluster"; exit 2; }

python -c "import openai" 2>/dev/null || {
  fail "openai package not installed; run: pip install openai"
  exit 2
}

# ── Phase 1: symlink the legacy ddg pool=50 html_cache so the new pipeline finds it
step "Symlink legacy ddg pool=50 html_cache → new ddg_*_serp50_top10 paths"
LEGACY="duckduckgo_Qwen2.5-72B-Instruct_serp50_top10/phase2/html_cache.tar.gz"
LEGACY_ABS="$GEODML_DATA_ROOT/data/runs/$LEGACY"
if [ ! -e "$LEGACY_ABS" ]; then
  fail "legacy html_cache not found at $LEGACY_ABS"
  fail "  -> need duckduckgo_Qwen2.5-72B-Instruct_serp50_top10/phase2/html_cache.tar.gz on disk"
  fail "  -> run: python scripts/sync_data.py  (or download_data.py)"
  exit 2
fi
for MODEL_TAG in Llama-3.3-70B-Instruct Qwen2.5-72B-Instruct; do
  TARGET_DIR="$GEODML_DATA_ROOT/data/runs/ddg_${MODEL_TAG}_serp50_top10/phase2"
  TARGET="$TARGET_DIR/html_cache.tar.gz"
  if [ -e "$TARGET" ] || [ -L "$TARGET" ]; then
    ok "${MODEL_TAG}: html_cache already in place"
    continue
  fi
  mkdir -p "$TARGET_DIR"
  ln -sfn "../../$LEGACY" "$TARGET"
  ok "${MODEL_TAG}: linked html_cache → legacy duckduckgo tarball"
done

# ── Phase 2: rerank — 4 cells (ddg × pool=50 × passage × 2 models)
if [ -z "${SKIP_RERANK:-}" ]; then
  step "Rerank — 4 missing cells via OpenAI-compatible API"
  for SPEC in "Llama-3.3-70B-Instruct|$LLAMA_MODEL_HF|$LLAMA_MODEL_API" \
              "Qwen2.5-72B-Instruct|$QWEN_MODEL_HF|$QWEN_MODEL_API"; do
    IFS='|' read -r TAG HF_ID API_ID <<<"$SPEC"
    for V in biased_passage neutral_passage; do
      RUN_DIR="$GEODML_DATA_ROOT/data/runs/ddg_${TAG}_serp50_top10_${V}"
      DONE_MARKER="$RUN_DIR/phase2/.done_rerank_${TAG}_${V}"
      if [ -f "$DONE_MARKER" ]; then
        ok "${TAG}/${V}: already done"
        continue
      fi
      go "${TAG}/${V}: rerank via API (model=$API_ID)"
      OPENAI_MODEL_OVERRIDE="$API_ID" \
      python -m interpretability.pipeline.rerank \
        --model "$HF_ID" \
        --engine ddg --pool 50 --variant "$V" \
        --backend openai --resume \
        ${MAX_KW:+--max-keywords "$MAX_KW"} \
        || fail "${TAG}/${V} failed (continuing)"
    done
  done
else
  skip "SKIP_RERANK=1 — leaving 4 ddg-passage rerank cells unfilled"
fi

# ── Phase 3: order_probe — same 4 cells × 2 seeds = 8 cells
if [ -z "${SKIP_ORDER_PROBE:-}" ]; then
  step "Order probe — 8 missing cells × 2 seeds via API"
  for SPEC in "Llama-3.3-70B-Instruct|$LLAMA_MODEL_HF|$LLAMA_MODEL_API" \
              "Qwen2.5-72B-Instruct|$QWEN_MODEL_HF|$QWEN_MODEL_API"; do
    IFS='|' read -r TAG HF_ID API_ID <<<"$SPEC"
    for V in biased_passage neutral_passage; do
      for SEED in 42 123; do
        OUT="$GEODML_DATA_ROOT/data/order_probe/ddg_${TAG}_serp50_top10_${V}_seed${SEED}.jsonl"
        DONE="$GEODML_DATA_ROOT/data/order_probe/.done_ddg_${TAG}_serp50_top10_${V}_seed${SEED}"
        if [ -f "$DONE" ] || [ -s "$OUT" ]; then
          ok "${TAG}/${V}/seed=${SEED}: already done"
          continue
        fi
        go "${TAG}/${V}/seed=${SEED}: order_probe via API"
        OPENAI_MODEL_OVERRIDE="$API_ID" \
        python -m interpretability.pipeline.order_probe \
          --model "$HF_ID" \
          --engine ddg --pool 50 --variant "$V" --seed "$SEED" \
          --backend openai --resume \
          ${MAX_KW:+--max-keywords "$MAX_KW"} \
          || fail "${TAG}/${V}/seed=${SEED} failed (continuing)"
      done
    done
  done

  step "Re-aggregate order_probe summary"
  if python -m interpretability.pipeline.order_probe_analyze 2>/dev/null; then
    ok "order_probe_summary.parquet refreshed"
  else
    skip "order_probe_analyze module not directly invokable; trying scripts/audit_status path"
    python -c "
from interpretability.pipeline.order_probe_analyze import main
import sys; sys.exit(main())
" 2>/dev/null || skip "(non-fatal — re-aggregate later)"
  fi
else
  skip "SKIP_ORDER_PROBE=1 — leaving 8 cells unfilled"
fi

# ── Phase 4: Stages B/C/D + figures + package (delegate to existing script)
if [ -z "${SKIP_BCD:-}" ]; then
  step "Stages B/C/D + figures + package"
  : "${FEATURES_DEVICE:=cpu}"
  export FEATURES_DEVICE
  bash scripts/continue_pipeline.sh
else
  skip "SKIP_BCD=1 — running audit only"
  python scripts/audit_pipeline.py
fi

step "Done"
cat <<NEXT
  What landed:
    - Stage A:  potentially 28/32 (24 already + 4 ddg-passage pool=50)
    - Stage A': potentially 56/64 (48 already + 8 ddg-passage pool=50)
    - Stages B / C / D: filled by continue_pipeline.sh
    - archives/local_results_<date>.zip: ready for upload
  What's still pending GPU:
    - 4 rerank cells (ddg × pool=20 × passage) — no ddg pool=20 HTML exists;
      not recoverable even with API
    - 8 order-probe cells (same blocker)
    - 6 probing cells (Stage F) — needs hidden states; resume on cluster
      when GPU access returns
NEXT
