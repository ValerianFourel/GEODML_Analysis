#!/usr/bin/env bash
# Build the RAG index for all (engine, pool) cells.
#
# Run THIS yourself in your terminal — Claude Code's sandbox blocks outbound
# OpenAI calls when the key is pasted in chat.
#
# Usage:
#   OPENAI_API_KEY=sk-... bash scripts/run_rag_embeddings.sh
#
# Optional env:
#   GEODML_DATA_ROOT       absolute path to your dataset root.
#                          default: ~/Hamburg/geodml-dataset
#   ENGINES                space-separated. default: "searxng ddg"
#   POOLS                  space-separated. default: "20 50"
#   EMBEDDING_MODEL        default: text-embedding-3-small
#   EMBEDDING_DIM          default: 1536  (Matryoshka — drop to 512 to save 3x disk)
#
# What this does:
#   1. Re-extracts body text (no 800-char cap) for ~44K unique URLs.
#   2. Chunks each page (~800-char chunks, 200-char overlap, ~5 chunks/page mean).
#   3. Embeds chunks AND keywords with OpenAI text-embedding-3-small.
#   4. Persists everything under data/rag_index/<engine>_top<pool>/ for
#      consumption by the rerank script (--variant biased_rag / neutral_rag).
#
# Cost: ~$1 total (~44M tokens × $0.02/1M).
# Wall: ~30-50 min total across all 4 cells.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY not set."
  echo "  Run as: OPENAI_API_KEY=sk-... bash scripts/run_rag_embeddings.sh"
  exit 2
fi

: "${GEODML_DATA_ROOT:=$HOME/Hamburg/geodml-dataset}"
: "${ENGINES:=searxng ddg}"
: "${POOLS:=20 50}"
: "${EMBEDDING_MODEL:=text-embedding-3-small}"
: "${EMBEDDING_DIM:=1536}"
export GEODML_DATA_ROOT OPENAI_API_KEY

read -r -a ENGINES_ARR <<<"$ENGINES"
read -r -a POOLS_ARR   <<<"$POOLS"

echo "──────────────────────────────────────────────"
echo "  GEODML_DATA_ROOT  = $GEODML_DATA_ROOT"
echo "  Engines           = ${ENGINES_ARR[*]}"
echo "  Pools             = ${POOLS_ARR[*]}"
echo "  Embedding model   = $EMBEDDING_MODEL  (dim=$EMBEDDING_DIM)"
echo "──────────────────────────────────────────────"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python

for ENGINE in "${ENGINES_ARR[@]}"; do
  for POOL in "${POOLS_ARR[@]}"; do
    echo
    echo "══════ $ENGINE / pool=$POOL ══════"
    "$PYTHON" -m interpretability.pipeline.build_rag_index \
      --engine "$ENGINE" --pool "$POOL" --resume \
      --embedding-model "$EMBEDDING_MODEL" \
      --embedding-dim "$EMBEDDING_DIM" || {
        echo "FAILED on $ENGINE/pool=$POOL — continuing with next cell"
      }
  done
done

echo
echo "──────────────────────────────────────────────"
echo "  All cells processed. Run the audit:"
echo "    GEODML_DATA_ROOT=$GEODML_DATA_ROOT $PYTHON -c '"
echo "    import json, pathlib"
echo "    for d in sorted((pathlib.Path(\"$GEODML_DATA_ROOT\")/\"data/rag_index\").iterdir()):"
echo "        meta = json.loads((d/\"meta.json\").read_text())"
echo "        print(f\"{d.name}: {meta[\\\"n_urls\\\"]} urls, {meta[\\\"n_chunks\\\"]} chunks, {meta[\\\"n_keywords\\\"]} kw\")'"
echo "──────────────────────────────────────────────"
