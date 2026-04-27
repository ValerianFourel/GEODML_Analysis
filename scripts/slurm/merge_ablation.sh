#!/usr/bin/env bash
# Concatenate the per-treatment ablation CSVs that the SLURM fan-out wrote
# into the canonical paths Figure A reads. Idempotent — safe to re-run.
#
# Sources:  interpretability/output/ablation_<TREATMENT>_<MODEL_TAG>/
#           ablation_results_{full,rw}.csv
# Targets:  interpretability/output/ablation_results_{full,rw}.csv

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO_ROOT/interpretability/output"

for SUF in _full _rw; do
  TARGET="$OUT/ablation_results${SUF}.csv"
  TMP="$TARGET.merge.tmp"
  : > "$TMP"
  first=1
  count=0
  for d in "$OUT"/ablation_*/; do
    f="$d/ablation_results${SUF}.csv"
    [ -f "$f" ] || continue
    if [ "$first" -eq 1 ]; then
      cat "$f" >> "$TMP"
      first=0
    else
      tail -n +2 "$f" >> "$TMP"
    fi
    count=$((count + 1))
  done
  if [ "$first" -eq 1 ]; then
    rm -f "$TMP"
    echo "[merge] no per-treatment CSVs for $SUF — skipping"
    continue
  fi
  mv "$TMP" "$TARGET"
  rows=$(($(wc -l < "$TARGET") - 1))
  echo "[merge] $TARGET  ($count partitions, $rows rows)"
done
