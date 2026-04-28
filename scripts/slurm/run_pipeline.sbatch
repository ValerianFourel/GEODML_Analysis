#!/bin/bash -l
# Single-submission orchestrator: chain rerank → features → merge → DML →
# ablation → saliency → probing → figures, with --dependency=afterok between
# stages. Each stage's individual chain_resubmit handles its own walltime
# recovery; this orchestrator just enforces stage ordering and a single
# operator entry point.
#
# Required exports: PROMPT_VARIANT (biased or neutral)
# Optional:         JUWELS_ACCOUNT, JUWELS_PROJECT, MODELS (default both),
#                   ENGINES (default 'searxng ddg'), POOLS (default '20 50'),
#                   SKIP_STAGES (e.g. 'features dml' to start from rerank+merge+interp)
#
# Submit (from repo root):
#   sbatch --account=$JUWELS_ACCOUNT \
#     --export=ALL,PROMPT_VARIANT=neutral \
#     scripts/slurm/run_pipeline.sbatch

#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --job-name=geodml-orchestrate
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail

# This job runs on a tiny CPU partition just to do submissions; no model/data
# needed. We still source _common.sh to pick up JUWELS_ACCOUNT/PROJECT etc.
# shellcheck source=scripts/slurm/_common.sh
source "$SLURM_SUBMIT_DIR/scripts/slurm/_common.sh"

: "${PROMPT_VARIANT:?PROMPT_VARIANT must be set: biased or neutral}"
: "${MODELS:=meta-llama/Llama-3.3-70B-Instruct Qwen/Qwen2.5-72B-Instruct}"
: "${ENGINES:=searxng ddg}"
: "${POOLS:=20 50}"
: "${SKIP_STAGES:=}"
: "${MAX_ATTEMPTS:=6}"

read -r -a MODELS_ARR  <<< "$MODELS"
read -r -a ENGINES_ARR <<< "$ENGINES"
read -r -a POOLS_ARR   <<< "$POOLS"
read -r -a SKIP_ARR    <<< "$SKIP_STAGES"

skip() { for s in "${SKIP_ARR[@]:-}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

ACCT_ARG=()
[ -n "${JUWELS_ACCOUNT:-}" ] && ACCT_ARG=(--account="$JUWELS_ACCOUNT")

submit() {
  local script="$1"; shift
  local jobname="$1"; shift
  local depend="$1"; shift
  local exports="ATTEMPT=1,MAX_ATTEMPTS=$MAX_ATTEMPTS,PROMPT_VARIANT=$PROMPT_VARIANT"
  [ -n "${JUWELS_PROJECT:-}" ] && exports="$exports,JUWELS_PROJECT=$JUWELS_PROJECT"
  for kv in "$@"; do exports="$exports,$kv"; done
  local cmd=(sbatch "${ACCT_ARG[@]}" --job-name="$jobname" --export="ALL,$exports")
  [ -n "$depend" ] && cmd+=(--dependency="afterok:$depend")
  cmd+=("$script")
  echo "[orchestrate] ${cmd[*]}"
  local out
  out=$("${cmd[@]}")
  echo "[orchestrate] -> $out"
  # Parse "Submitted batch job <ID>" -> <ID>.
  echo "$out" | awk '/Submitted batch job/ { print $4 }'
}

# ── 1. Stage A: rerank, fan out per (engine, model, pool) ────────────────────
RERANK_DEPS=()
if ! skip rerank; then
  for ENGINE in "${ENGINES_ARR[@]}"; do
    for MODEL in "${MODELS_ARR[@]}"; do
      TAG="${MODEL##*/}"
      for POOL in "${POOLS_ARR[@]}"; do
        JID=$(submit scripts/slurm/run_rerank.sbatch \
          "rerank-${TAG}-${ENGINE}-${POOL}-${PROMPT_VARIANT}" "" \
          "MODEL=$MODEL" "ENGINE=$ENGINE" "POOL=$POOL")
        [ -n "$JID" ] && RERANK_DEPS+=("$JID")
      done
    done
  done
fi
RERANK_DEP_STR=$(IFS=:; echo "${RERANK_DEPS[*]:-}")

# ── 2. Stage B: features, fan out per (engine, pool) ─────────────────────────
FEAT_DEPS=()
if ! skip features; then
  for ENGINE in "${ENGINES_ARR[@]}"; do
    for POOL in "${POOLS_ARR[@]}"; do
      JID=$(submit scripts/slurm/run_features.sbatch \
        "features-${ENGINE}-${POOL}" "$RERANK_DEP_STR" \
        "ENGINE=$ENGINE" "POOL=$POOL")
      [ -n "$JID" ] && FEAT_DEPS+=("$JID")
    done
  done
fi
FEAT_DEP_STR=$(IFS=:; echo "${FEAT_DEPS[*]:-}")
ALL_PRE_DEPS=$(IFS=:; deps=(); deps+=("${RERANK_DEPS[@]:-}"); deps+=("${FEAT_DEPS[@]:-}"); echo "${deps[*]:-}")

# ── 3. Stages C + D: merge + DML on the login-ish CPU partition ──────────────
DML_DEP=""
if ! skip dml; then
  # Wrap merge + dml in a small one-off CPU job so the chain dependency is clean.
  JID=$(submit scripts/slurm/run_dml.sbatch \
    "dml-${PROMPT_VARIANT}" "$ALL_PRE_DEPS")
  DML_DEP="$JID"
fi

# Note on Stage C (merge): build_main_table.py is fast (~30 s pure pandas)
# and runs *inside* run_dml.sbatch via a one-line wrapper at job start.
# See operator runbook docs/pipeline-port.md for the exact incantation.

# ── 4. Stage F: re-run interpretability (ablation/saliency/probing) on the new
#     full_experiment_data_{variant}.parquet. PROMPT_VARIANT propagates via
#     --export=ALL so the rerank prompt switches in interpretability.utils.
INTERP_DEPS=()
if ! skip ablation; then
  for MODEL in "${MODELS_ARR[@]}"; do
    TAG="${MODEL##*/}"
    for T in T7_source_earned T5_topical_comp T3_structured_data_new \
             T2a_question_headings T6_freshness T1b_stats_density; do
      JID=$(submit scripts/slurm/run_ablation.sbatch \
        "abl-${TAG}-${T}-${PROMPT_VARIANT}" "$DML_DEP" \
        "MODEL=$MODEL" "TREATMENT=$T")
      [ -n "$JID" ] && INTERP_DEPS+=("$JID")
    done
  done
fi
if ! skip saliency; then
  for MODEL in "${MODELS_ARR[@]}"; do
    TAG="${MODEL##*/}"
    for FRAME in full robust_winners; do
      JID=$(submit scripts/slurm/run_saliency.sbatch \
        "sal-${TAG}-${FRAME}-${PROMPT_VARIANT}" "$DML_DEP" \
        "MODEL=$MODEL" "FRAME=$FRAME")
      [ -n "$JID" ] && INTERP_DEPS+=("$JID")
    done
  done
fi
if ! skip probing; then
  for MODEL in "${MODELS_ARR[@]}"; do
    TAG="${MODEL##*/}"
    JID=$(submit scripts/slurm/run_probing.sbatch \
      "prob-${TAG}-${PROMPT_VARIANT}" "$DML_DEP" \
      "MODEL=$MODEL")
    [ -n "$JID" ] && INTERP_DEPS+=("$JID")
  done
fi

echo "[orchestrate] all stages submitted for PROMPT_VARIANT=$PROMPT_VARIANT"
echo "[orchestrate]   rerank   jobs: ${#RERANK_DEPS[@]}"
echo "[orchestrate]   features jobs: ${#FEAT_DEPS[@]}"
echo "[orchestrate]   dml      job : ${DML_DEP:-(skipped)}"
echo "[orchestrate]   interp   jobs: ${#INTERP_DEPS[@]}"
exit 0
