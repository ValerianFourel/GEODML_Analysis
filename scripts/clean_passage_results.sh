#!/usr/bin/env bash
# Wipe all polluted *_passage results so we can re-run them with the SHA-256
# fix in place. The results that get deleted were generated with the old
# MD5-based html_cache lookup → every passage was an empty string, so they're
# functionally identical to snippet variants and should be discarded.
#
# What gets deleted (all under $GEODML_DATA_ROOT):
#   - data/runs/<engine>_<Model>_serp<N>_top10_<*_passage>/phase2/
#       keywords.jsonl, .rerank_ckpt.json, .done_rerank_*
#   - data/order_probe/<engine>_<Model>_serp<N>_top10_<*_passage>_seed*.jsonl
#       and matching .done_* / *_ckpt.json
#   - data/passages/passages_*.parquet  (cache, will be rebuilt with sha256)
#   - data/order_probe/order_probe_summary.parquet  (will be regenerated)
#
# Survives untouched:
#   - All snippet variants (biased, neutral) — they don't use html_cache
#   - All Stage F outputs (ablation, saliency, weights, probing)
#   - The cluster snapshot's html_cache.tar.gz files
#   - Local Mac html_cache directories
#   - Phase0 SERP files
#
# DRY-RUN by default: prints what it would delete. Pass FORCE=1 to actually delete.

set -uo pipefail

: "${GEODML_DATA_ROOT:?Set GEODML_DATA_ROOT to your dataset root}"
: "${FORCE:=}"

step() { printf '\n\033[1m══════ %s ══════\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
go()   { printf '  \033[36m→\033[0m %s\n' "$*"; }

if [ -z "$FORCE" ]; then
  step "DRY RUN — set FORCE=1 to actually delete"
else
  step "DELETING polluted *_passage results"
fi

n_files=0
del() {
  local p="$1"
  if [ -e "$p" ] || [ -L "$p" ]; then
    if [ -n "$FORCE" ]; then
      rm -rf "$p"
    fi
    n_files=$((n_files + 1))
    echo "  $p"
  fi
}

# 1. Per-cell rerank outputs (16 cells: 2 engines × 2 models × 2 pools × 2 variants)
step "1/4  Rerank cells"
for ENG in searxng ddg; do
  for MODEL in Llama-3.3-70B-Instruct Qwen2.5-72B-Instruct; do
    for POOL in 20 50; do
      for V in biased_passage neutral_passage; do
        D="$GEODML_DATA_ROOT/data/runs/${ENG}_${MODEL}_serp${POOL}_top10_${V}/phase2"
        [ -d "$D" ] || continue
        del "$D/keywords.jsonl"
        del "$D/rankings.csv"
        del "$D/.rerank_ckpt.json"
        for f in "$D"/.done_rerank_*; do del "$f"; done 2>/dev/null
      done
    done
  done
done

# 2. Order probe jsonls (32 cells = same × 2 seeds)
step "2/4  Order probe outputs"
OP="$GEODML_DATA_ROOT/data/order_probe"
for ENG in searxng ddg; do
  for MODEL in Llama-3.3-70B-Instruct Qwen2.5-72B-Instruct; do
    for POOL in 20 50; do
      for V in biased_passage neutral_passage; do
        for SEED in 42 123; do
          BASE="${ENG}_${MODEL}_serp${POOL}_top10_${V}_seed${SEED}"
          del "$OP/${BASE}.jsonl"
          del "$OP/.done_${BASE}"
          del "$OP/.${BASE}_ckpt.json"
        done
      done
    done
  done
done

# 3. Passage extraction caches (built with empty entries from MD5 lookups)
step "3/4  Passage caches"
for f in "$GEODML_DATA_ROOT"/data/passages/passages_*.parquet; do del "$f"; done 2>/dev/null

# 4. Aggregated order_probe_summary (will be regenerated next time)
step "4/4  Aggregated order_probe summary"
del "$GEODML_DATA_ROOT/data/order_probe/order_probe_summary.parquet"

step "Summary"
echo "  $n_files items"
if [ -z "$FORCE" ]; then
  echo
  echo "  This was a DRY RUN. To actually delete, run:"
  echo "    FORCE=1 GEODML_DATA_ROOT=$GEODML_DATA_ROOT bash scripts/clean_passage_results.sh"
fi
