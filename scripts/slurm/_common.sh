# Shared SLURM env setup for GEODML interpretability jobs on JUWELS Booster.
# Source me from each *.sbatch wrapper:  source scripts/slurm/_common.sh
#
# Boots the project venv, loads HF + dataset caches off $SCRATCH, and switches
# transformers/HF Hub into offline mode (compute nodes have no internet on
# JUWELS Booster — see https://apps.fz-juelich.de/jsc/hps/juwels/batchsystem.html).
#
# Required env (set by the wrapper, not here):
#   SLURM_SUBMIT_DIR  — repo root, where .venv/, .env, geodml_data/ live
# Optional:
#   JUWELS_PROJECT    — passed to `jutil env activate -p <project>`
#   HF_HOME           — override the default $SCRATCH/hf_cache
#   GEODML_DATA_ROOT  — override the default $SCRATCH/geodml_data

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Project env (sets PROJECT, SCRATCH, etc. on JSC systems).
if [ -n "${JUWELS_PROJECT:-}" ]; then
  jutil env activate -p "$JUWELS_PROJECT"
fi

# Module stack. Stages/2024 is the JSC default at time of writing; bump if the
# site rolls forward. `module spider <name>` shows availability.
module load Stages/2024 GCC Python CUDA 2>&1 | sed 's/^/[modules] /' || true

# Activate the project venv created on the login node.
if [ ! -f "$SLURM_SUBMIT_DIR/.venv/bin/activate" ]; then
  echo "[common] ERROR: $SLURM_SUBMIT_DIR/.venv missing — create it on the login node first."
  exit 99
fi
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

# Load .env (HF_TOKEN, HF_DATASET_REPO, model lists, ...).
if [ -f "$SLURM_SUBMIT_DIR/.env" ]; then
  set -o allexport
  # shellcheck disable=SC1091
  source "$SLURM_SUBMIT_DIR/.env"
  set +o allexport
fi

# Caches on fast scratch. Offline because JUWELS Booster compute nodes
# have no outbound HTTP — pre-populate $HF_HOME on the login node.
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GEODML_DATA_ROOT="${GEODML_DATA_ROOT:-$SCRATCH/geodml_data}"

# Make sklearn / numpy stay inside our 48-cpu allocation.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-48}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$SLURM_SUBMIT_DIR/logs"

echo "[common] node=$(hostname) jobid=${SLURM_JOB_ID:-?} attempt=${ATTEMPT:-1}/${MAX_ATTEMPTS:-6}"
echo "[common] HF_HOME=$HF_HOME"
echo "[common] GEODML_DATA_ROOT=$GEODML_DATA_ROOT"
echo "[common] visible GPUs:"
nvidia-smi -L 2>&1 | sed 's/^/[common]   /' || echo "[common] nvidia-smi unavailable"

# Helper: chain a follow-up job that will pick up via --resume. Pass the path
# to the current sbatch script as $1 and any extra --export key=value pairs
# as $2..$N. Caller decides whether to call this (typically only on rc != 0).
chain_resubmit() {
  local script="$1"; shift
  local extra=""
  for kv in "$@"; do
    extra="${extra:+$extra,}$kv"
  done
  local attempt="${ATTEMPT:-1}"
  local max="${MAX_ATTEMPTS:-6}"
  if [ "$attempt" -ge "$max" ]; then
    echo "[chain] reached MAX_ATTEMPTS=$max — stopping. Investigate before resubmitting."
    return 1
  fi
  local next=$((attempt + 1))
  echo "[chain] queueing attempt $next/$max with --dependency=afterany:$SLURM_JOB_ID"
  # JSC's submit filter requires --account on every sbatch even when ALL is
  # forwarded — env exports don't satisfy it. Pull it from JUWELS_ACCOUNT.
  local account_arg=()
  if [ -n "${JUWELS_ACCOUNT:-}" ]; then
    account_arg=(--account="$JUWELS_ACCOUNT")
  else
    echo "[chain] WARNING: JUWELS_ACCOUNT unset; sbatch will likely reject the submission."
  fi
  sbatch \
    "${account_arg[@]}" \
    --dependency=afterany:"$SLURM_JOB_ID" \
    --export="ALL,ATTEMPT=$next,MAX_ATTEMPTS=$max${extra:+,$extra}" \
    "$script"
}
