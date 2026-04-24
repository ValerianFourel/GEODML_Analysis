#!/usr/bin/env bash
# End-to-end driver for the GEODML interpretability pipeline.
# Designed for a single-GPU box with HF_TOKEN set in .env.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -f .env ]; then
  set -o allexport
  # shellcheck disable=SC1091
  source .env
  set +o allexport
fi

: "${HF_TOKEN:?HF_TOKEN not set. Copy .env.example to .env first.}"

mkdir -p interpretability/output/plots

SAMPLE_ABL="${SAMPLE_ABL:-500}"
SAMPLE_SAL="${SAMPLE_SAL:-200}"
SAMPLE_PROBE="${SAMPLE_PROBE:-2000}"

echo "==> 0/5  Download dataset (skip if already present)"
python scripts/download_data.py

echo "==> 1/5  Sanity-check DML table"
python scripts/sanity_check.py || echo "(mismatch logged; continuing)"

echo "==> 2/5  Option 1 — Ablation (HF Inference API, ~2-4 h)"
python -m interpretability.ablation --sample-n "$SAMPLE_ABL" --resume

echo "==> 3/5  Extract HTML caches (needed for Options 2 & 3)"
python scripts/download_data.py --no-download --extract-html

echo "==> 4/5  Option 2 — Saliency (local GPU, ~1-2 h)"
python -m interpretability.saliency --sample-n "$SAMPLE_SAL" --resume

echo "==> 5/5  Option 3 — Probing (local GPU, ~30-60 min)"
python -m interpretability.probing --sample-n "$SAMPLE_PROBE" --resume

echo "==> figures"
python -m interpretability.make_figures

echo "DONE. Outputs in interpretability/output/"
