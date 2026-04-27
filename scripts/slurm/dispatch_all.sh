#!/usr/bin/env bash
# Dispatch the full GEODML interpretability pipeline on JUWELS Booster.
# Submits up to 20 sbatch jobs that fan out across the queue:
#
#   2 models  ×  6 ablation treatments     = 12 ablation jobs
#   2 models  ×  2 frames (saliency)       =  4 saliency jobs
#   2 models  ×  1 (probing, both frames)  =  2 probing jobs
#   2 models  ×  1 (weights)               =  2 weight-analysis jobs
#
# Every job sets --frame both (or runs one frame explicitly), --sample-n huge
# so all eligible keywords are taken, and --resume. Every job self-resubmits
# with --dependency=afterany:<jobid> on walltime expiry or non-zero exit, up
# to MAX_ATTEMPTS times — so a 36 h ablation run survives a 24 h walltime cap.
#
# Required env (in .env or your shell):
#   JUWELS_ACCOUNT    — SLURM accounting budget, passed to every sbatch
#   JUWELS_PROJECT    — JSC project for `jutil env activate`
#   HF_TOKEN          — already used by the python scripts
#
# Usage:
#   ./scripts/slurm/dispatch_all.sh                # full real run
#   ./scripts/slurm/dispatch_all.sh --dry-run      # print sbatch commands only
#   ./scripts/slurm/dispatch_all.sh --smoke        # develbooster, small N
#   ./scripts/slurm/dispatch_all.sh --only ablation
#   ./scripts/slurm/dispatch_all.sh --models meta-llama/Llama-3.3-70B-Instruct
#
# After all chains complete, run scripts/slurm/merge_ablation.sh to consolidate
# the per-treatment ablation CSVs into the canonical
# interpretability/output/ablation_results_{full,rw}.csv files.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ----- defaults ---------------------------------------------------------------

DEFAULT_MODELS=(
  "meta-llama/Llama-3.3-70B-Instruct"
  "Qwen/Qwen2.5-72B-Instruct"
)
TREATMENTS=(
  T7_source_earned
  T5_topical_comp
  T3_structured_data_new
  T2a_question_headings
  T6_freshness
  T1b_stats_density
)
FRAMES=(full robust_winners)

DRY_RUN=0
SMOKE=0
ONLY=""
MODELS=("${DEFAULT_MODELS[@]}")
PARTITION=""
TIME_OVERRIDE=""
SAMPLE_N=""
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --smoke)
      SMOKE=1
      PARTITION="develbooster"
      TIME_OVERRIDE="01:30:00"
      SAMPLE_N=50
      MAX_ATTEMPTS=2
      shift ;;
    --only) ONLY="$2"; shift 2 ;;
    --models) IFS=',' read -r -a MODELS <<< "$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --time) TIME_OVERRIDE="$2"; shift 2 ;;
    --sample-n) SAMPLE_N="$2"; shift 2 ;;
    --help|-h)
      sed -n '2,30p' "$0"
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ -z "${JUWELS_ACCOUNT:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
  set -o allexport; . "$REPO_ROOT/.env"; set +o allexport
fi
: "${JUWELS_ACCOUNT:?JUWELS_ACCOUNT must be set in .env or your shell}"

mkdir -p logs

# ----- one helper: build & emit one sbatch invocation -------------------------

emit() {
  local script="$1"; shift
  local jobname="$1"; shift
  # remaining args are key=value exports

  local exports="ATTEMPT=1,MAX_ATTEMPTS=$MAX_ATTEMPTS"
  [ -n "${JUWELS_PROJECT:-}" ] && exports="$exports,JUWELS_PROJECT=$JUWELS_PROJECT"
  [ -n "$SAMPLE_N" ]            && exports="$exports,SAMPLE_N=$SAMPLE_N"
  for kv in "$@"; do exports="$exports,$kv"; done

  local cmd=(
    sbatch
    --account="$JUWELS_ACCOUNT"
    --job-name="$jobname"
    --export="ALL,$exports"
  )
  [ -n "$PARTITION" ]     && cmd+=(--partition="$PARTITION")
  [ -n "$TIME_OVERRIDE" ] && cmd+=(--time="$TIME_OVERRIDE")
  cmd+=("$script")

  if [ "$DRY_RUN" -eq 1 ]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
}

# ----- dispatch --------------------------------------------------------------

want() {
  [ -z "$ONLY" ] && return 0
  [ "$ONLY" = "$1" ]
}

if want ablation; then
  echo "[dispatch] ablation: ${#MODELS[@]} models × ${#TREATMENTS[@]} treatments"
  for MODEL in "${MODELS[@]}"; do
    TAG="${MODEL##*/}"
    for T in "${TREATMENTS[@]}"; do
      emit scripts/slurm/run_ablation.sbatch "abl-${TAG}-${T}" \
        "MODEL=$MODEL" "TREATMENT=$T"
    done
  done
fi

if want saliency; then
  echo "[dispatch] saliency: ${#MODELS[@]} models × ${#FRAMES[@]} frames"
  for MODEL in "${MODELS[@]}"; do
    TAG="${MODEL##*/}"
    for F in "${FRAMES[@]}"; do
      emit scripts/slurm/run_saliency.sbatch "sal-${TAG}-${F}" \
        "MODEL=$MODEL" "FRAME=$F"
    done
  done
fi

if want probing; then
  echo "[dispatch] probing: ${#MODELS[@]} models (frame=both)"
  for MODEL in "${MODELS[@]}"; do
    TAG="${MODEL##*/}"
    emit scripts/slurm/run_probing.sbatch "prob-${TAG}" \
      "MODEL=$MODEL"
  done
fi

if want weights; then
  echo "[dispatch] weights: ${#MODELS[@]} models"
  for MODEL in "${MODELS[@]}"; do
    TAG="${MODEL##*/}"
    emit scripts/slurm/run_weights.sbatch "wgt-${TAG}" \
      "MODEL=$MODEL"
  done
fi

echo "[dispatch] done."
if [ "$SMOKE" -eq 1 ]; then
  echo "[dispatch] SMOKE mode: develbooster, time=$TIME_OVERRIDE, sample_n=$SAMPLE_N"
  echo "[dispatch] reminder: develbooster is capped at 4 submitted / 2 running."
fi
