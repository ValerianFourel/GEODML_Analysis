#!/usr/bin/env bash
# One-time migration: rename existing un-suffixed interp output dirs so they
# carry an explicit variant tag. Pre-port outputs were all biased, so we
# default to suffixing them with _biased. Idempotent — safe to re-run; skips
# anything already moved.
#
# Affected directories (under interpretability/output/):
#     ablation_<TREATMENT>_<MODEL>           -> ablation_<TREATMENT>_<MODEL>_biased
#     saliency_<MODEL>                       -> saliency_<MODEL>_biased
#     probing_<MODEL>                        -> probing_<MODEL>_biased
#     weights_<MODEL>                        -> weights_<MODEL>_biased
#
# Usage:
#     scripts/migrate_interp_outputs_to_variant.sh           # dry-run
#     scripts/migrate_interp_outputs_to_variant.sh --apply   # actually move
#     scripts/migrate_interp_outputs_to_variant.sh --apply --variant biased
#                                                             # explicit variant
#
# After running with --apply, re-run scripts/audit_status.py and
# scripts/audit_pipeline.py to confirm nothing was missed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/interpretability/output"

VARIANT="biased"
APPLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --apply)   APPLY=1; shift ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,22p' "$0"
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ ! -d "$OUT_DIR" ]; then
    echo "[migrate] $OUT_DIR does not exist; nothing to migrate."
    exit 0
fi

declare -i moved=0 skipped=0 nochange=0

migrate_one() {
    local dir="$1"
    local base
    base="$(basename "$dir")"

    # If it already ends in _biased / _neutral / etc., skip.
    case "$base" in
        *_biased|*_neutral) skipped+=1; return ;;
    esac

    local target="${dir}_${VARIANT}"

    if [ -e "$target" ]; then
        echo "[migrate] SKIP $base -> ${base}_${VARIANT} (target already exists)"
        skipped+=1
        return
    fi

    if [ "$APPLY" -eq 1 ]; then
        mv "$dir" "$target"
        echo "[migrate] MV   $base -> ${base}_${VARIANT}"
    else
        echo "[migrate] DRY  $base -> ${base}_${VARIANT}"
    fi
    moved+=1
}

shopt -s nullglob
for prefix in ablation saliency probing weights; do
    for d in "$OUT_DIR/${prefix}"_*/; do
        [ -d "$d" ] || continue
        # Strip trailing slash for cleaner output.
        d="${d%/}"
        migrate_one "$d"
    done
done

echo "[migrate] summary: moved=$moved skipped=$skipped (apply=$APPLY variant=$VARIANT)"
if [ "$APPLY" -eq 0 ] && [ "$moved" -gt 0 ]; then
    echo "[migrate] this was a dry run. Re-run with --apply to actually move."
fi
