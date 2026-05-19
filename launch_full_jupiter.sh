#!/bin/bash
# Full JUPITER campaign dispatcher: 32 rerank + 64 order_probe = 96 jobs.
# Re-runs every (model × engine × pool × variant) cell at bf16-full on GH200,
# plus order_probe at two seeds for each cell, to standardize precision across
# the GEODML DML experiment.
#
# Usage:
#   ./launch_full_jupiter.sh                  # archive existing outputs, then submit all 96
#   DRY_RUN=1 ./launch_full_jupiter.sh        # print sbatch invocations only
#   SKIP_CLEAR=1 ./launch_full_jupiter.sh     # do NOT archive — let --resume pick up in-progress cells
#
# SKIP_CLEAR is the safe choice for re-runs after a fix: rerank.py's --resume
# skips keywords that already have records, so existing keywords.jsonl files
# accelerate the campaign rather than being thrown away.

set -uo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a

DRY="${DRY_RUN:-0}"
SKIP_CLEAR="${SKIP_CLEAR:-0}"
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

submit() {
  local jobname=$1; shift
  local cmd=(sbatch
    --account=scifi --partition=booster --time=05:55:00
    --nodes=1 --ntasks=1 --cpus-per-task=48 --gres=gpu:4
    --job-name="$jobname"
    --output="logs/${jobname}-%j.out"
    --error="logs/${jobname}-%j.err"
    "$@")
  if [ "$DRY" = "1" ]; then
    printf '  %q ' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
}

MODELS=(meta-llama/Llama-3.3-70B-Instruct Qwen/Qwen2.5-72B-Instruct)
ENGINES=(searxng ddg)
POOLS=(20 50)
VARIANTS=(biased neutral biased_rag neutral_rag)
SEEDS=(42 123)

if [ "$DRY" != "1" ] && [ "$SKIP_CLEAR" != "1" ]; then
  echo "[clear] archiving previous outputs (set SKIP_CLEAR=1 to skip)..."
  for V in "${VARIANTS[@]}"; do
    for D in $GEODML_DATA_ROOT/data/runs/*_top10_${V}/phase2; do
      [ -f "$D/keywords.jsonl" ] && mv "$D/keywords.jsonl" "$D/keywords.jsonl.bak-${TS}"
      rm -f "$D/.rerank_ckpt.json" "$D/rankings.csv"
    done
    for S in "${SEEDS[@]}"; do
      for F in $GEODML_DATA_ROOT/data/order_probe/*_${V}_seed${S}.jsonl; do
        [ -f "$F" ] && mv "$F" "${F%.jsonl}.bak-${TS}.jsonl"
      done
    done
  done
elif [ "$SKIP_CLEAR" = "1" ]; then
  echo "[clear] SKIP_CLEAR=1 — keeping existing outputs; --resume will pick up where each cell left off."
fi

echo "[rerank] 32 jobs"
for M in "${MODELS[@]}"; do TAG="${M##*/}"
for E in "${ENGINES[@]}"; do
for P in "${POOLS[@]}"; do
for V in "${VARIANTS[@]}"; do
  submit "rerank-${TAG}-${E}-${P}-${V}" \
    --export=ALL,MODEL=$M,ENGINE=$E,POOL=$P,PROMPT_VARIANT=$V,LOCAL_PRECISION=full \
    scripts/slurm/run_rerank.sbatch
done; done; done; done

echo "[order_probe] 64 jobs"
for M in "${MODELS[@]}"; do TAG="${M##*/}"
for E in "${ENGINES[@]}"; do
for P in "${POOLS[@]}"; do
for V in "${VARIANTS[@]}"; do
for S in "${SEEDS[@]}"; do
  submit "op-${TAG}-${E}-${P}-${V}-s${S}" \
    --export=ALL,MODEL=$M,ENGINE=$E,POOL=$P,PROMPT_VARIANT=$V,SEED=$S,LOCAL_PRECISION=full \
    scripts/slurm/run_order_probe.sbatch
done; done; done; done; done

echo "[done] all submitted at $(date)"
if [ "$DRY" != "1" ]; then
  squeue -u $USER --format='%.10i %.32j %.2t %.10M' | head -10
  echo "[total] $(squeue -u $USER -h | wc -l)"
fi
