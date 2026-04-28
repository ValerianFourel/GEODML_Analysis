#!/usr/bin/env bash
# Concatenate the per-treatment ablation CSVs that the SLURM fan-out wrote,
# now variant-aware. Idempotent — safe to re-run.
#
# Sources:  interpretability/output/ablation_<TREATMENT>_<MODEL_TAG>_<VARIANT>/
#           ablation_results_{full,rw}.csv
# Targets:  interpretability/output/ablation_results_{full,rw}_<VARIANT>.csv
#
#           For backward compat, when VARIANT=biased we ALSO write the
#           legacy un-suffixed paths (ablation_results_full.csv) so existing
#           consumers (figure_a, audit_status legacy) still work.
#
# Usage:
#     scripts/slurm/merge_ablation.sh                    # both variants if present
#     scripts/slurm/merge_ablation.sh --variant biased   # one variant
#     scripts/slurm/merge_ablation.sh --variant neutral

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO_ROOT/interpretability/output"

VARIANTS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --variant) VARIANTS+=("$2"); shift 2 ;;
        --help|-h) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Default: merge every variant we find directories for. Uses a portable
# dedup pattern so this stays compatible with bash 3.2 (macOS).
if [ "${#VARIANTS[@]}" -eq 0 ]; then
    shopt -s nullglob
    seen_biased=0
    seen_neutral=0
    for d in "$OUT"/ablation_*_biased/; do
        [ -d "$d" ] || continue
        seen_biased=1
        break
    done
    for d in "$OUT"/ablation_*_neutral/; do
        [ -d "$d" ] || continue
        seen_neutral=1
        break
    done
    [ "$seen_biased" -eq 1 ]  && VARIANTS+=("biased")
    [ "$seen_neutral" -eq 1 ] && VARIANTS+=("neutral")
    if [ "${#VARIANTS[@]}" -eq 0 ]; then
        echo "[merge] no variant-suffixed ablation_*/ dirs found under $OUT"
        echo "[merge] (run scripts/migrate_interp_outputs_to_variant.sh first if you have legacy outputs)"
        exit 0
    fi
fi

merge_one_variant() {
    local variant="$1"
    for SUF in _full _rw; do
        local TARGET="$OUT/ablation_results${SUF}_${variant}.csv"
        local TMP="$TARGET.merge.tmp"
        : > "$TMP"
        local first=1 count=0
        for d in "$OUT"/ablation_*_"${variant}"/; do
            local f="$d/ablation_results${SUF}.csv"
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
            echo "[merge] [$variant] no per-treatment CSVs for $SUF — skipping"
            continue
        fi
        mv "$TMP" "$TARGET"
        local rows
        rows=$(($(wc -l < "$TARGET") - 1))
        echo "[merge] [$variant] $TARGET  ($count partitions, $rows rows)"

        # Backwards compat: biased also gets written to the legacy path.
        if [ "$variant" = "biased" ]; then
            local LEGACY="$OUT/ablation_results${SUF}.csv"
            cp "$TARGET" "$LEGACY"
            echo "[merge] [$variant] legacy alias -> $LEGACY"
        fi
    done
}

for v in "${VARIANTS[@]}"; do
    case "$v" in
        biased|neutral) ;;
        *) echo "[merge] unknown variant: $v (expected biased|neutral)" >&2; exit 2 ;;
    esac
    merge_one_variant "$v"
done
