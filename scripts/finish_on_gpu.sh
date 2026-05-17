#!/usr/bin/env bash
# Finish Stage A (rerank), Stage A' (order_probe), and Stage F gap-fills on
# JUWELS Booster cluster GPUs, under API-equivalent conditions (bf16,
# no quantization).
#
# Structural analog of finish_via_api.sh, but submits sbatch jobs instead of
# running inline. Uses dispatch_all.sh as the single submission surface so
# slurm export/dispatch logic stays in one place.
#
# Why this exists (work-log 2026-05-08 + 2026-05-17):
#   - Snippet cells (biased, neutral) were originally run with LocalRanker's
#     4-bit nf4 quantization. RAG cells were started via HF Inference API at
#     full precision. The two arms are not directly comparable.
#   - User decision: full reconciliation. All Stage A/A' work runs in
#     LOCAL_PRECISION=full (bf16) on the cluster. Snippet is re-done; RAG
#     pending cells (26 of 32) are finished; existing RAG cells (already
#     full-precision via API) are kept unless INCLUDE_RAG_REDO=1.
#
# Required env (or .env / shell):
#   GEODML_DATA_ROOT   abs path to dataset dir (data/runs/, data/order_probe/, ...)
#   JUWELS_ACCOUNT     slurm accounting budget (passed to every sbatch)
#   JUWELS_PROJECT     (optional) jutil project activation tag
#
# Optional env:
#   MAX_KW=400                 target keyword count per cell (enables skip-guard)
#   LOCAL_PRECISION=full|4bit  default "full"
#   INCLUDE_RAG_REDO=1         also re-run the RAG rerank/order_probe in bf16
#   INCLUDE_F_GAPS=1           also queue Stage F probing for `neutral` + all
#                              four Stage F methods for `_rag` variants
#   ENGINES="searxng ddg"      restrict engine axis
#   POOLS="20 50"              restrict pool axis
#   SEEDS="42 123"             restrict order_probe seed axis
#   MODELS="meta-llama/Llama-3.3-70B-Instruct Qwen/Qwen2.5-72B-Instruct"
#   DRY_RUN=1                  forward --dry-run to dispatch_all.sh
#
# Usage examples:
#   GEODML_DATA_ROOT=$SCRATCH/geodml_data ./scripts/finish_on_gpu.sh
#   MAX_KW=400 INCLUDE_RAG_REDO=1 ./scripts/finish_on_gpu.sh
#   DRY_RUN=1 ./scripts/finish_on_gpu.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

step() { printf '\n\033[1m══════ %s ══════\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
go()   { printf '  \033[36m→\033[0m %s\n' "$*"; }
skip() { printf '  \033[33m⊘\033[0m %s\n' "$*"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$*"; }

# ── Load .env if present, fill required env (caller's env wins over .env) ───
# We snapshot any env vars the caller already set, source .env to fill in
# the rest, then restore the caller's values. This keeps .env as a *default*
# source rather than an override (otherwise GEODML_DATA_ROOT=./geodml_data
# in .env would clobber a $SCRATCH/geodml_data passed on the command line).
if [ -f "$REPO_ROOT/.env" ]; then
  declare -A _PRESET=()
  for _v in GEODML_DATA_ROOT JUWELS_ACCOUNT JUWELS_PROJECT HF_TOKEN \
            OPENAI_API_KEY OPENAI_BASE_URL LOCAL_PRECISION MAX_KW; do
    if [ -n "${!_v:-}" ]; then
      _PRESET[$_v]="${!_v}"
    fi
  done
  set -o allexport
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.env"
  set +o allexport
  for _v in "${!_PRESET[@]}"; do
    export "$_v=${_PRESET[$_v]}"
  done
  unset _PRESET _v
fi

: "${GEODML_DATA_ROOT:?GEODML_DATA_ROOT must be set (e.g. \$SCRATCH/geodml_data)}"
: "${JUWELS_ACCOUNT:?JUWELS_ACCOUNT must be set in .env or shell}"
export GEODML_DATA_ROOT JUWELS_ACCOUNT
[ -n "${JUWELS_PROJECT:-}" ] && export JUWELS_PROJECT

# ── Knobs with sensible defaults ─────────────────────────────────────────────
# MAX_KW="" / "0" / "all" / "unlimited" / "none" means: process every keyword in
#   the cell, no cap. That makes skip_if_at_max a no-op so the rerank loop runs
#   to completion on every keyword. rerank.py --resume still skips
#   already-done keywords, so re-runs are idempotent.
: "${MAX_KW:=400}"
case "$(echo "$MAX_KW" | tr '[:upper:]' '[:lower:]')" in
  ""|0|all|unlimited|none) MAX_KW=0 ;;
esac
: "${LOCAL_PRECISION:=full}"
: "${INCLUDE_RAG_REDO:=}"
: "${INCLUDE_F_GAPS:=}"
: "${ENGINES:=searxng ddg}"
: "${POOLS:=20 50}"
: "${SEEDS:=42 123}"
: "${MODELS:=meta-llama/Llama-3.3-70B-Instruct Qwen/Qwen2.5-72B-Instruct}"
: "${DRY_RUN:=}"

# Comma-joined forms for dispatch_all.sh's CSV-style flags.
ENGINES_CSV="${ENGINES// /,}"
POOLS_CSV="${POOLS// /,}"
SEEDS_CSV="${SEEDS// /,}"
MODELS_CSV="${MODELS// /,}"

DISPATCH="$REPO_ROOT/scripts/slurm/dispatch_all.sh"
[ -x "$DISPATCH" ] || { fail "$DISPATCH not executable"; exit 2; }

DRY_FLAG=()
[ -n "$DRY_RUN" ] && DRY_FLAG=(--dry-run)

# Exported to every child sbatch via dispatch_all.sh's emit helper. MAX_KW
# becomes MAX_KEYWORDS in the slurm scripts (kept as-is for consistency with
# finish_via_api.sh wording). If MAX_KW=0 (unlimited mode), unset MAX_KEYWORDS
# so the python --max-keywords flag is not passed at all — rerank.py will then
# loop through every keyword in the cell's SERP parquet.
export LOCAL_PRECISION
if [ "$MAX_KW" = "0" ]; then
  unset MAX_KEYWORDS
else
  export MAX_KEYWORDS="$MAX_KW"
fi

step "Pre-flight"
echo "  REPO_ROOT          = $REPO_ROOT"
echo "  GEODML_DATA_ROOT   = $GEODML_DATA_ROOT"
echo "  JUWELS_ACCOUNT     = $JUWELS_ACCOUNT"
echo "  LOCAL_PRECISION    = $LOCAL_PRECISION   (bf16 if 'full', nf4 if '4bit')"
if [ "$MAX_KW" = "0" ]; then
  echo "  MAX_KW (target)    = UNLIMITED          (no skip-guard; process every keyword)"
else
  echo "  MAX_KW (target)    = $MAX_KW            (skip cells already at ≥)"
fi
echo "  MODELS             = $MODELS"
echo "  ENGINES            = $ENGINES"
echo "  POOLS              = $POOLS"
echo "  SEEDS              = $SEEDS"
echo "  INCLUDE_RAG_REDO   = ${INCLUDE_RAG_REDO:-0}   (re-run RAG rerank+op in bf16)"
echo "  INCLUDE_F_GAPS     = ${INCLUDE_F_GAPS:-0}   (Stage F: probing/neutral + all/rag)"
echo "  DRY_RUN            = ${DRY_RUN:-0}"

# Verify config drift hasn't broken the API equivalence assumption.
python -c "
from interpretability.pipeline import config as C
assert C.LLM_TEMPERATURE == 0.1, C.LLM_TEMPERATURE
assert C.LLM_MAX_TOKENS == 500, C.LLM_MAX_TOKENS
print(f'  LLM config         = temperature={C.LLM_TEMPERATURE}, max_tokens={C.LLM_MAX_TOKENS}')
" || { fail "config.py drifted — fix before launching"; exit 2; }

# ── Phase 1.5: RAG index sanity (cluster cannot build it — needs OpenAI API)
# All four (engine, pool) cells should already have meta.json + chunk_embeddings.npy.
step "RAG index sanity check"
rag_missing=0
for ENGINE in $ENGINES; do
  for POOL in $POOLS; do
    META="$GEODML_DATA_ROOT/data/rag_index/${ENGINE}_top${POOL}/meta.json"
    EMB="$GEODML_DATA_ROOT/data/rag_index/${ENGINE}_top${POOL}/chunk_embeddings.npy"
    if [ -f "$META" ] && [ -f "$EMB" ]; then
      ok "rag_index ${ENGINE}/pool=${POOL}: present"
    else
      skip "rag_index ${ENGINE}/pool=${POOL}: MISSING ($META or $EMB)"
      rag_missing=$((rag_missing + 1))
    fi
  done
done
if [ "$rag_missing" -gt 0 ]; then
  fail "$rag_missing RAG cells missing — _rag variants will fail."
  fail "Build the index from Mac (needs OpenAI API key):"
  fail "  cd ~/Hamburg/GEODML_Analysis && \\"
  fail "  for E in $ENGINES; do for P in $POOLS; do \\"
  fail "    python -m interpretability.pipeline.build_rag_index --engine \$E --pool \$P --resume; \\"
  fail "  done; done"
  fail "Then rsync data/rag_index/ to the cluster."
  fail "Continuing — snippet-only phases will still work."
fi

# ── Phase 2: rerank (snippet redo; optional RAG redo) ────────────────────────
SNIPPET_VARIANTS=(biased neutral)
RAG_VARIANTS=(biased_rag neutral_rag)

step "Phase 2 — rerank (snippet redo)"
echo "  Submitting $((${#SNIPPET_VARIANTS[@]} * $(echo $MODELS | wc -w) * $(echo $ENGINES | wc -w) * $(echo $POOLS | wc -w))) cells via dispatch_all.sh --only rerank."
echo "  Each sbatch skip_if_at_max guards against re-running cells already at MAX_KW=$MAX_KW."
for V in "${SNIPPET_VARIANTS[@]}"; do
  go "rerank variant=$V → dispatch_all.sh"
  "$DISPATCH" "${DRY_FLAG[@]}" \
    --only rerank \
    --variant "$V" \
    --models "$MODELS_CSV" \
    --engines "$ENGINES_CSV" \
    --pools "$POOLS_CSV" \
    --precision "$LOCAL_PRECISION"
done

if [ -n "$INCLUDE_RAG_REDO" ]; then
  step "Phase 2 — rerank (RAG redo, INCLUDE_RAG_REDO=1)"
  for V in "${RAG_VARIANTS[@]}"; do
    go "rerank variant=$V → dispatch_all.sh"
    "$DISPATCH" "${DRY_FLAG[@]}" \
      --only rerank \
      --variant "$V" \
      --models "$MODELS_CSV" \
      --engines "$ENGINES_CSV" \
      --pools "$POOLS_CSV" \
      --precision "$LOCAL_PRECISION"
  done
else
  skip "RAG rerank kept as-is (set INCLUDE_RAG_REDO=1 to overwrite with bf16 runs)"
fi

# ── Phase 3: order_probe (snippet redo + RAG finish) ─────────────────────────
# Snippet cells were 4-bit; re-do under bf16 so seed=42/123 match the snippet
# Stage A re-do. RAG cells: 6 already done via API + 1 partial + 25 pending —
# the skip_if_at_max guard inside run_order_probe.sbatch handles the boundary
# so we can safely submit the full grid.

if [ -n "$INCLUDE_RAG_REDO" ]; then
  ALL_OP_VARIANTS=("${SNIPPET_VARIANTS[@]}" "${RAG_VARIANTS[@]}")
else
  ALL_OP_VARIANTS=("${SNIPPET_VARIANTS[@]}" "${RAG_VARIANTS[@]}")
  # Note: even without RAG_REDO we DO submit RAG order_probe — the API-done
  # cells are skipped by the skip_if_at_max guard (they're at MAX_KW already);
  # only the 26 pending cells actually run.
fi

OP_VARIANTS_CSV=$(IFS=,; echo "${ALL_OP_VARIANTS[*]}")

step "Phase 3 — order_probe (snippet redo + RAG finish)"
echo "  variants: ${ALL_OP_VARIANTS[*]}"
echo "  seeds:    $SEEDS"
echo "  Skip guard auto-skips cells already at MAX_KW=$MAX_KW (e.g. the 6 API'd RAG cells)."
go "dispatch_all.sh --only order_probe"
"$DISPATCH" "${DRY_FLAG[@]}" \
  --only order_probe \
  --order-probe-variants "$OP_VARIANTS_CSV" \
  --models "$MODELS_CSV" \
  --engines "$ENGINES_CSV" \
  --pools "$POOLS_CSV" \
  --seeds "$SEEDS_CSV" \
  --precision "$LOCAL_PRECISION"

# ── Phase 4: Stage F gap-fills (optional) ────────────────────────────────────
if [ -n "$INCLUDE_F_GAPS" ]; then
  step "Phase 4 — Stage F gap fills"
  # 4a) probing missing for `neutral` (one-armed mechanistic story risk —
  #     see docs/long-term-project-arc.md §7.1).
  for V in neutral; do
    go "probing variant=$V → dispatch_all.sh"
    "$DISPATCH" "${DRY_FLAG[@]}" \
      --only probing \
      --variant "$V" \
      --models "$MODELS_CSV"
  done
  # 4b) full Stage F for _rag variants.
  for V in biased_rag neutral_rag; do
    for METHOD in ablation saliency probing weights; do
      go "$METHOD variant=$V → dispatch_all.sh"
      "$DISPATCH" "${DRY_FLAG[@]}" \
        --only "$METHOD" \
        --variant "$V" \
        --models "$MODELS_CSV"
    done
  done
else
  skip "Stage F gaps unchanged (set INCLUDE_F_GAPS=1 to queue probing/neutral + all/_rag)"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
step "Submitted"
cat <<NEXT
  Monitor:
    squeue -u \$USER --format='%.10i %.32j %.2t %.10M %.20S' | head
    tail -f logs/*.out

  Pre-flight skip in action: cells already at MAX_KW=$MAX_KW exit 0 immediately
    via skip_if_at_max in scripts/slurm/_common.sh (no GPU time wasted).

  After all chains finish, on the login node:
    python scripts/audit_status.py    # expect 100% across all 6 variants
    bash   scripts/continue_pipeline.sh  # Stage B (cached) + C/D × 4 variants + figures

  Re-sync back to Mac:
    rsync -avz --partial juwels:\$SCRATCH/geodml_data/data/ \$HOME/Hamburg/geodml-dataset/data/

  Notes:
    - Snippet rerank runs at bf16 (~2× slower than archived 4-bit data).
      Archive old 4-bit data first if you want to compare:
        mv \$GEODML_DATA_ROOT/data/runs/{searxng,ddg}_*_top10_{biased,neutral} \\
           \$GEODML_DATA_ROOT/archives/snippet_4bit_\$(date +%Y%m%d)/
    - If a Qwen-72B bf16 cell OOMs on 4×80 GB, the chain_resubmit will
      retry up to MAX_ATTEMPTS=6; if it persists, drop --top-k-rag to 2
      for _rag variants or split across nodes.
NEXT
