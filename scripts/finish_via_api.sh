#!/usr/bin/env bash
# Finish Stage A 32/32 + Stage A' 64/64 + Stages B/C/D, all without GPU.
#
# Designed for the case where you've lost cluster GPU access. Uses local
# html_cache directories from the upstream experiment (LOCAL_HTML_SOURCE)
# and an inference API for the LLM. Two backend options:
#
#   BACKEND=hf       (default; cheapest path)
#                    HuggingFace InferenceClient. Reads HF_TOKEN. Routes
#                    automatically to whichever provider hosts the model
#                    cheapest right now (Together, DeepInfra, Sambanova,
#                    Fireworks, …). One token, no new account, free tier
#                    fits a smoke test, pay-as-you-go scales.
#
#   BACKEND=openai   OpenAI-compatible endpoint at OPENAI_BASE_URL
#                    (DeepInfra / Together / Fireworks direct). Use this
#                    if HF route refuses to host the model or you want a
#                    specific provider's pricing.
#
# What this fills:
#   - 8 rerank cells: ddg × {pool=20, pool=50} × {biased_passage,
#     neutral_passage} × {Llama, Qwen} — Stage A 24/32 → 32/32
#   - 16 order-probe cells: same 8 × 2 seeds — Stage A' 48/64 → 64/64
#   - Stage B (features) for all engines — variant-agnostic, deterministic
#   - Stage C (merge) for all 4 variants
#   - Stage D (DML) for all 4 variants
#   - Paper figures
#
# What stays unfilled (need cluster GPU later):
#   - 6 probing cells (Stage F: probing 2/8 → 8/8) — needs raw model
#     hidden states; resume on cluster when GPU access returns
#
# Required env:
#   GEODML_DATA_ROOT  abs path to the (unzipped) data dir
#   plus one of:
#     HF_TOKEN          if BACKEND=hf (default) — your HF read token works
#     OPENAI_API_KEY    if BACKEND=openai
#     OPENAI_BASE_URL   if BACKEND=openai (or use PROVIDER=...)
#
# Optional env:
#   BACKEND             "hf" (default) / "openai"
#   PROVIDER            "deepinfra" / "together" / "fireworks" presets for openai
#   LOCAL_HTML_SOURCE   path holding the legacy {searxng,duckduckgo}_<Model>
#                       _serp<N>_top10/html_cache/ directories. Default
#                       auto-detect via known paths.
#   SKIP_RERANK / SKIP_ORDER_PROBE / SKIP_BCD  set to "1" to skip a phase
#   FEATURES_DEVICE     "cpu" (default) / "mps" / "cuda"
#   MAX_KW              cap keywords per cell for smoke testing

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

step() { printf '\n\033[1m══════ %s ══════\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
go()   { printf '  \033[36m→\033[0m %s\n' "$*"; }
skip() { printf '  \033[33m⊘\033[0m %s\n' "$*"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$*"; }

[ -n "${GEODML_DATA_ROOT:-}" ] || { fail "GEODML_DATA_ROOT not set"; exit 2; }
export GEODML_DATA_ROOT

# ── Provider preset ──────────────────────────────────────────────────────────
case "${PROVIDER:-}" in
  deepinfra)
    : "${OPENAI_BASE_URL:=https://api.deepinfra.com/v1/openai}"
    LLAMA_API_ID="meta-llama/Meta-Llama-3.1-70B-Instruct"
    QWEN_API_ID="Qwen/Qwen2.5-72B-Instruct"
    ;;
  together)
    : "${OPENAI_BASE_URL:=https://api.together.xyz/v1}"
    LLAMA_API_ID="meta-llama/Llama-3.3-70B-Instruct-Turbo"
    QWEN_API_ID="Qwen/Qwen2.5-72B-Instruct-Turbo"
    ;;
  fireworks)
    : "${OPENAI_BASE_URL:=https://api.fireworks.ai/inference/v1}"
    LLAMA_API_ID="accounts/fireworks/models/llama-v3p3-70b-instruct"
    QWEN_API_ID="accounts/fireworks/models/qwen2p5-72b-instruct"
    ;;
  *)
    LLAMA_API_ID="${LLAMA_API_ID:-meta-llama/Llama-3.3-70B-Instruct}"
    QWEN_API_ID="${QWEN_API_ID:-Qwen/Qwen2.5-72B-Instruct}"
    ;;
esac

LLAMA_HF_ID="meta-llama/Llama-3.3-70B-Instruct"
QWEN_HF_ID="Qwen/Qwen2.5-72B-Instruct"

# ── Auto-detect LOCAL_HTML_SOURCE ────────────────────────────────────────────
if [ -z "${LOCAL_HTML_SOURCE:-}" ]; then
  for cand in \
    "$HOME/Hamburg/GEODML/paperSizeExperiment/output" \
    "$HOME/GEODML/paperSizeExperiment/output" \
    "/Users/valerianfourel/Hamburg/GEODML/paperSizeExperiment/output"; do
    if [ -d "$cand" ] && \
       [ -d "$cand/duckduckgo_Qwen2.5-72B-Instruct_serp20_top10/html_cache" ]; then
      LOCAL_HTML_SOURCE="$cand"
      break
    fi
  done
fi

# ── Backend selection ────────────────────────────────────────────────────────
: "${BACKEND:=hf}"
case "$BACKEND" in
  hf|api)
    BACKEND_ARG="api"
    if [ -z "${SKIP_RERANK:-}" ] || [ -z "${SKIP_ORDER_PROBE:-}" ]; then
      [ -n "${HF_TOKEN:-}" ] || {
        fail "BACKEND=hf requires HF_TOKEN (or use BACKEND=openai). Set HF_TOKEN=hf_..."
        exit 2
      }
      export HF_TOKEN
    fi
    ;;
  openai)
    BACKEND_ARG="openai"
    if [ -z "${SKIP_RERANK:-}" ] || [ -z "${SKIP_ORDER_PROBE:-}" ]; then
      [ -n "${OPENAI_API_KEY:-}" ] || { fail "OPENAI_API_KEY not set"; exit 2; }
      [ -n "${OPENAI_BASE_URL:-}" ] || { fail "OPENAI_BASE_URL not set (or PROVIDER=...)"; exit 2; }
      export OPENAI_API_KEY OPENAI_BASE_URL
    fi
    ;;
  *)
    fail "BACKEND must be 'hf' (HF Inference) or 'openai' (OpenAI-compat). Got '$BACKEND'."
    exit 2
    ;;
esac

step "Pre-flight"
echo "  GEODML_DATA_ROOT   = $GEODML_DATA_ROOT"
echo "  LOCAL_HTML_SOURCE  = ${LOCAL_HTML_SOURCE:-(none — only HF tarballs will be used)}"
echo "  BACKEND            = $BACKEND  (rerank/order_probe --backend=$BACKEND_ARG)"
if [ "$BACKEND_ARG" = "api" ]; then
  echo "  HF_TOKEN           = ${HF_TOKEN:0:8}…   (routed via huggingface_hub.InferenceClient)"
  echo "  Models (HF ids)    = $LLAMA_HF_ID, $QWEN_HF_ID"
else
  echo "  OPENAI_BASE_URL    = ${OPENAI_BASE_URL:-(unset)}"
  echo "  Llama API id       = $LLAMA_API_ID"
  echo "  Qwen  API id       = $QWEN_API_ID"
fi

python -c "
from interpretability.pipeline import config as C
assert C.LLM_TEMPERATURE == 0.1, C.LLM_TEMPERATURE
assert C.LLM_MAX_TOKENS == 500, C.LLM_MAX_TOKENS
print(f'  LLM config         = temperature={C.LLM_TEMPERATURE}, max_tokens={C.LLM_MAX_TOKENS} (matches cluster)')
" || { fail "config.py drifted from cluster"; exit 2; }

if [ "$BACKEND_ARG" = "openai" ]; then
  python -c "import openai" 2>/dev/null || {
    fail "openai package not installed; run: pip install openai"
    exit 2
  }
fi

# ── Phase 1: symlink local html_caches into the new pipeline's expected paths
# The new pipeline expects:  data/runs/<engine>_<Model>_serp<N>_top10/phase2/html_cache
# The legacy local layout is: $LOCAL_HTML_SOURCE/<engine_legacy>_<Model>_serp<N>_top10/html_cache
# Rename: duckduckgo -> ddg
step "Symlink local html_caches into new pipeline paths"
linked=0; already=0; missing=0
if [ -n "${LOCAL_HTML_SOURCE:-}" ] && [ -d "$LOCAL_HTML_SOURCE" ]; then
  for ENGINE_LEGACY in searxng duckduckgo; do
    [ "$ENGINE_LEGACY" = "duckduckgo" ] && ENGINE_NEW="ddg" || ENGINE_NEW="$ENGINE_LEGACY"
    for MODEL in Llama-3.3-70B-Instruct Qwen2.5-72B-Instruct; do
      for POOL in 20 50; do
        SRC="$LOCAL_HTML_SOURCE/${ENGINE_LEGACY}_${MODEL}_serp${POOL}_top10/html_cache"
        DEST_DIR="$GEODML_DATA_ROOT/data/runs/${ENGINE_NEW}_${MODEL}_serp${POOL}_top10/phase2"
        DEST="$DEST_DIR/html_cache"
        cell_label="${ENGINE_NEW}/${MODEL}/pool=${POOL}"
        if [ ! -d "$SRC" ]; then
          # Not strictly missing — the pipeline may have a tarball at DEST/.tar.gz
          if [ -e "$DEST_DIR/html_cache.tar.gz" ] || [ -d "$DEST" ] || [ -L "$DEST" ]; then
            ok "$cell_label: already cached (tarball or extracted)"
          else
            skip "$cell_label: no local source, no HF tarball — passages unavailable"
            missing=$((missing + 1))
          fi
          continue
        fi
        if [ -e "$DEST" ] || [ -L "$DEST" ]; then
          ok "$cell_label: already linked"
          already=$((already + 1))
          continue
        fi
        mkdir -p "$DEST_DIR"
        ln -sfn "$SRC" "$DEST"
        n=$(ls "$SRC" 2>/dev/null | wc -l | tr -d ' ')
        ok "$cell_label: linked → $SRC ($n files)"
        linked=$((linked + 1))
      done
    done
  done
else
  skip "LOCAL_HTML_SOURCE not found — only cells with HF tarballs will work"
fi
echo "  summary: linked=$linked, already=$already, missing=$missing"

# ── Phase 2: rerank — 8 cells (ddg × {pool=20, pool=50} × passage × 2 models)
if [ -z "${SKIP_RERANK:-}" ]; then
  step "Rerank — 8 missing ddg×passage cells via OpenAI-compatible API"
  for SPEC in "Llama-3.3-70B-Instruct|$LLAMA_HF_ID|$LLAMA_API_ID" \
              "Qwen2.5-72B-Instruct|$QWEN_HF_ID|$QWEN_API_ID"; do
    IFS='|' read -r TAG HF_ID API_ID <<<"$SPEC"
    for POOL in 20 50; do
      for V in biased_passage neutral_passage; do
        cell_label="${TAG}/pool=${POOL}/${V}"
        # Verify HTML is reachable for this cell
        HTML_DIR="$GEODML_DATA_ROOT/data/runs/ddg_${TAG}_serp${POOL}_top10/phase2"
        if [ ! -e "$HTML_DIR/html_cache.tar.gz" ] && [ ! -d "$HTML_DIR/html_cache" ] && [ ! -L "$HTML_DIR/html_cache" ]; then
          skip "$cell_label: no html_cache (Phase 1 didn't link it)"
          continue
        fi
        RUN_DIR="$GEODML_DATA_ROOT/data/runs/ddg_${TAG}_serp${POOL}_top10_${V}"
        DONE_MARKER="$RUN_DIR/phase2/.done_rerank_${TAG}_${V}"
        if [ -f "$DONE_MARKER" ]; then
          ok "$cell_label: already done"
          continue
        fi
        if [ "$BACKEND_ARG" = "api" ]; then
          go "$cell_label: rerank via HF Inference (model=$HF_ID)"
          python -m interpretability.pipeline.rerank \
            --model "$HF_ID" \
            --engine ddg --pool "$POOL" --variant "$V" \
            --backend api --resume \
            ${MAX_KW:+--max-keywords "$MAX_KW"} \
            || fail "$cell_label failed (continuing)"
        else
          go "$cell_label: rerank via OpenAI-compat (model=$API_ID)"
          OPENAI_MODEL_OVERRIDE="$API_ID" \
          python -m interpretability.pipeline.rerank \
            --model "$HF_ID" \
            --engine ddg --pool "$POOL" --variant "$V" \
            --backend openai --resume \
            ${MAX_KW:+--max-keywords "$MAX_KW"} \
            || fail "$cell_label failed (continuing)"
        fi
      done
    done
  done
else
  skip "SKIP_RERANK=1 — leaving 8 ddg-passage rerank cells unfilled"
fi

# ── Phase 3: order_probe — same 8 cells × 2 seeds = 16 cells
if [ -z "${SKIP_ORDER_PROBE:-}" ]; then
  step "Order probe — 16 missing ddg×passage cells (8 cells × 2 seeds) via API"
  for SPEC in "Llama-3.3-70B-Instruct|$LLAMA_HF_ID|$LLAMA_API_ID" \
              "Qwen2.5-72B-Instruct|$QWEN_HF_ID|$QWEN_API_ID"; do
    IFS='|' read -r TAG HF_ID API_ID <<<"$SPEC"
    for POOL in 20 50; do
      for V in biased_passage neutral_passage; do
        for SEED in 42 123; do
          cell_label="${TAG}/pool=${POOL}/${V}/seed=${SEED}"
          HTML_DIR="$GEODML_DATA_ROOT/data/runs/ddg_${TAG}_serp${POOL}_top10/phase2"
          if [ ! -e "$HTML_DIR/html_cache.tar.gz" ] && [ ! -d "$HTML_DIR/html_cache" ] && [ ! -L "$HTML_DIR/html_cache" ]; then
            skip "$cell_label: no html_cache"
            continue
          fi
          OUT="$GEODML_DATA_ROOT/data/order_probe/ddg_${TAG}_serp${POOL}_top10_${V}_seed${SEED}.jsonl"
          DONE="$GEODML_DATA_ROOT/data/order_probe/.done_ddg_${TAG}_serp${POOL}_top10_${V}_seed${SEED}"
          if [ -f "$DONE" ] || [ -s "$OUT" ]; then
            ok "$cell_label: already done"
            continue
          fi
          if [ "$BACKEND_ARG" = "api" ]; then
            go "$cell_label: order_probe via HF Inference"
            python -m interpretability.pipeline.order_probe \
              --model "$HF_ID" \
              --engine ddg --pool "$POOL" --variant "$V" --seed "$SEED" \
              --backend api --resume \
              ${MAX_KW:+--max-keywords "$MAX_KW"} \
              || fail "$cell_label failed (continuing)"
          else
            go "$cell_label: order_probe via OpenAI-compat"
            OPENAI_MODEL_OVERRIDE="$API_ID" \
            python -m interpretability.pipeline.order_probe \
              --model "$HF_ID" \
              --engine ddg --pool "$POOL" --variant "$V" --seed "$SEED" \
              --backend openai --resume \
              ${MAX_KW:+--max-keywords "$MAX_KW"} \
              || fail "$cell_label failed (continuing)"
          fi
        done
      done
    done
  done

  step "Re-aggregate order_probe summary"
  python -c "
from interpretability.pipeline import order_probe_analyze as M
import sys
sys.exit(M.main() if hasattr(M, 'main') else 0)
" 2>/dev/null \
    && ok "order_probe_summary.parquet refreshed" \
    || skip "(non-fatal — re-aggregate later if needed)"
else
  skip "SKIP_ORDER_PROBE=1 — leaving 16 cells unfilled"
fi

# ── Phase 4: Stages B/C/D + figures + package
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
  Expected end state:
    Stage A   32/32   (was 24/32 — all 8 ddg×passage cells now via API)
    Stage A'  64/64   (was 48/64)
    Stage B    4/4    (now includes ddg via local html_cache; was 2/4)
    Stage C    4/4
    Stage D    4/4
    Stage F   74/80   (probing still 2/8 — needs GPU)

  Upload:
    The packaging step inside continue_pipeline.sh produced
      archives/local_results_<date>.zip
    and printed the hf upload command. Run it with a write-scoped HF token.

  Stage F probing (when GPU access returns):
    See PROVENANCE.txt inside the zip for the sbatch invocation.
NEXT
