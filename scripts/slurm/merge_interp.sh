#!/usr/bin/env bash
# Consolidate per-(model, variant) interp outputs into top-level merged CSVs
# that the figures can read.
#
# Sources:
#   ablation_<TREATMENT>_<MODEL>_<VARIANT>/ablation_results_{full,rw}.csv
#   saliency_<MODEL>_<VARIANT>/saliency_summary_{full,rw}.csv
#   saliency_<MODEL>_<VARIANT>/saliency_scores_{full,rw}.csv
#   probing_<MODEL>_<VARIANT>/probing_results.csv
#   weights_<MODEL>_<VARIANT>/{logit_lens,attention_heads}.csv
#
# Targets:
#   ablation_results_{full,rw}_<VARIANT>.csv      (+ legacy alias for biased)
#   saliency_summary_{full,rw}_<VARIANT>.csv
#   saliency_scores_{full,rw}_<VARIANT>.csv
#   probing_results_<VARIANT>.csv
#   logit_lens_<VARIANT>.csv
#   attention_heads_<VARIANT>.csv
#
# Usage:
#     scripts/slurm/merge_interp.sh                    # both variants if present
#     scripts/slurm/merge_interp.sh --variant biased
#     scripts/slurm/merge_interp.sh --stage saliency   # only saliency
#
# Idempotent.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO_ROOT/interpretability/output"

VARIANTS=()
STAGES=()

while [ $# -gt 0 ]; do
    case "$1" in
        --variant) VARIANTS+=("$2"); shift 2 ;;
        --stage)   STAGES+=("$2"); shift 2 ;;
        --help|-h) sed -n '2,28p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

[ "${#VARIANTS[@]}" -eq 0 ] && VARIANTS=(biased neutral)
[ "${#STAGES[@]}"   -eq 0 ] && STAGES=(ablation saliency probing weights)

shopt -s nullglob

# Concatenate every CSV matching a glob into TARGET, deduping the header.
# args: TARGET, csv_basename (under each src dir), <glob of source dirs>
concat_csvs() {
    local target="$1"; shift
    local csv_base="$1"; shift
    local label="$1"; shift
    # remaining args are source-dir patterns
    local tmp="$target.merge.tmp"
    : > "$tmp"
    local first=1 count=0
    for pattern in "$@"; do
        for d in $pattern; do
            [ -d "$d" ] || continue
            local f="$d/$csv_base"
            [ -f "$f" ] || continue
            if [ "$first" -eq 1 ]; then
                cat "$f" >> "$tmp"
                first=0
            else
                tail -n +2 "$f" >> "$tmp"
            fi
            count=$((count + 1))
        done
    done
    if [ "$first" -eq 1 ]; then
        rm -f "$tmp"
        echo "[merge_interp] $label: no inputs — skipping"
        return 1
    fi
    mv "$tmp" "$target"
    local rows
    rows=$(($(wc -l < "$target") - 1))
    echo "[merge_interp] $label  ($count partitions, $rows rows) -> $(basename "$target")"
    return 0
}

want() {
    local needle="$1"
    for s in "${STAGES[@]}"; do
        [ "$s" = "$needle" ] && return 0
    done
    return 1
}

merge_ablation_variant() {
    local v="$1"
    for SUF in _full _rw; do
        concat_csvs "$OUT/ablation_results${SUF}_${v}.csv" \
                    "ablation_results${SUF}.csv" \
                    "ablation/$v/$SUF" \
                    "$OUT/ablation_*_${v}/" || true
        # legacy alias for biased: ablation_results_full.csv / ablation_results_rw.csv
        if [ "$v" = "biased" ] && [ -f "$OUT/ablation_results${SUF}_${v}.csv" ]; then
            cp "$OUT/ablation_results${SUF}_${v}.csv" "$OUT/ablation_results${SUF}.csv"
            echo "[merge_interp] ablation/$v/$SUF legacy alias -> ablation_results${SUF}.csv"
        fi
    done
}

merge_saliency_variant() {
    local v="$1"
    for SUF in _full _rw; do
        for STEM in saliency_summary saliency_scores; do
            concat_csvs "$OUT/${STEM}${SUF}_${v}.csv" \
                        "${STEM}${SUF}.csv" \
                        "saliency/$v/${STEM}${SUF}" \
                        "$OUT/saliency_*_${v}/" || true
            if [ "$v" = "biased" ] && [ -f "$OUT/${STEM}${SUF}_${v}.csv" ]; then
                cp "$OUT/${STEM}${SUF}_${v}.csv" "$OUT/${STEM}${SUF}.csv"
                echo "[merge_interp] saliency/$v/${STEM}${SUF} legacy alias -> ${STEM}${SUF}.csv"
            fi
        done
    done
}

merge_probing_variant() {
    local v="$1"
    concat_csvs "$OUT/probing_results_${v}.csv" \
                "probing_results.csv" \
                "probing/$v" \
                "$OUT/probing_*_${v}/" || true
    if [ "$v" = "biased" ] && [ -f "$OUT/probing_results_${v}.csv" ]; then
        cp "$OUT/probing_results_${v}.csv" "$OUT/probing_results.csv"
        echo "[merge_interp] probing/$v legacy alias -> probing_results.csv"
    fi
}

merge_weights_variant() {
    local v="$1"
    for STEM in logit_lens attention_heads; do
        concat_csvs "$OUT/${STEM}_${v}.csv" \
                    "${STEM}.csv" \
                    "weights/$v/${STEM}" \
                    "$OUT/weights_*_${v}/" || true
        if [ "$v" = "biased" ] && [ -f "$OUT/${STEM}_${v}.csv" ]; then
            cp "$OUT/${STEM}_${v}.csv" "$OUT/${STEM}.csv"
            echo "[merge_interp] weights/$v/${STEM} legacy alias -> ${STEM}.csv"
        fi
    done
}

for v in "${VARIANTS[@]}"; do
    case "$v" in
        biased|neutral) ;;
        *) echo "[merge_interp] unknown variant: $v" >&2; exit 2 ;;
    esac
    echo "[merge_interp] === variant=$v ==="
    want ablation && merge_ablation_variant "$v"
    want saliency && merge_saliency_variant "$v"
    want probing  && merge_probing_variant "$v"
    want weights  && merge_weights_variant  "$v"
done
