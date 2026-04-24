# GEODML Interpretability

Three mechanistic-interpretability analyses that validate and extend the DML
causal estimates from the GEODML paper (EMNLP 2026, deadline 2026-05-25).

All input data lives in the HF dataset
[`ValerianFourel/geodml-papersize`](https://huggingface.co/datasets/ValerianFourel/geodml-papersize)
(private). Everything here is **compute** — no raw data is committed.

---

## Quick start

```bash
# 1. Python env
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# On the GPU box only:
pip install "torch>=2.4" "transformers>=4.45" "accelerate>=0.34" "bitsandbytes>=0.43"

# 2. Config
cp .env.example .env          # then paste your HF read token into .env

# 3. Data
python scripts/download_data.py                 # ~3.5 GB
python scripts/download_data.py --extract-html  # +24 GB, only needed for Options 2 & 3

# 4. Sanity check — reproduces the POOLED DML table from the paper
python scripts/sanity_check.py

# 5. Run the three analyses (or use run_all.sh)
python -m interpretability.ablation --sample-n 500    # HF Inference API, ~2-4 h
python -m interpretability.saliency --sample-n 200    # local GPU, ~1-2 h
python -m interpretability.probing  --sample-n 2000   # local GPU, ~30-60 min

# 6. Paper figures
python -m interpretability.make_figures
```

### Air-gapped clusters (e.g. Jülich)

All three analyses run with zero outbound HTTP — the HF bundle already
contains the full SERP snapshots (`data/serp/phase0_*.parquet`) and every
page's cached HTML (`data/runs/*/phase2/html_cache.tar.gz`). For ablation,
switch the re-ranker to a locally-loaded model:

```bash
# Stage the dataset on a machine with internet, rsync to the cluster, then:
ABLATION_BACKEND=local ./run_all.sh
# or:
python -m interpretability.ablation --sample-n 500 --backend local \
    --models meta-llama/Llama-3.1-8B-Instruct
```

`--backend local` reuses the same 4-bit quantization path as saliency and
probing, so ablation now fits in ~10 GB VRAM. No `HF_TOKEN` needed beyond
the initial dataset download.

---

## Layout

```
scripts/
  download_data.py        Pull the HF dataset, optionally extract HTML caches.
  sanity_check.py         Confirm POOLED DML coefficients match the paper.
interpretability/
  utils.py                Shared: data loader, HTML loader, prompt builder,
                          HF Inference client, checkpointer.
  ablation.py             Option 1 — input ablation via HF Inference API.
  saliency.py             Option 2 — gradient x input saliency on local GPU.
  probing.py              Option 3 — per-layer logistic probes on local GPU.
  make_figures.py         Produces figure_a/b/c.png for the paper.
  output/                 CSVs, plots, checkpoints (gitignored).
run_all.sh                End-to-end driver with sensible defaults.
```

---

## What each analysis produces

| File | What it is | Feeds figure |
|---|---|---|
| `interpretability/output/ablation_results.csv` | per-(keyword, url, treatment) baseline rank, ablated rank, delta | Figure A |
| `interpretability/output/saliency_scores.csv`  | per-token saliency, tagged with which treatment the token belongs to | Figure B |
| `interpretability/output/saliency_summary.csv` | per-treatment mean saliency ratio (treatment vs other tokens) | Figure B |
| `interpretability/output/probing_results.csv`  | per-(treatment, layer, pooling) probe accuracy + ROC-AUC | Figure C |

Paper figures land in `interpretability/output/plots/` with filenames
`figure_a_ablation.png`, `figure_b_saliency.png`, `figure_c_probing.png`.

---

## GPU requirements

- **Ablation (Option 1):** no GPU. Uses the HF Inference API; needs `HF_TOKEN`
  and rate-limit tolerance.
- **Saliency (Option 2):** 1x GPU with ≥12 GB VRAM (8B in 4-bit ≈ 6 GB model
  weights + activations + gradients). Tested on A100-40G and RTX 4090.
- **Probing (Option 3):** 1x GPU with ≥10 GB VRAM. No backward pass needed —
  hidden-state extraction is cheap.

Everything auto-detects CUDA; saliency and probing fall back to CPU with a
loud warning and an ETA that reflects the slowdown.

---

## Treatment → column mapping (cheat sheet)

| DML label | main-table column | direction on `rank_delta` |
|---|---|---|
| `T7_source_earned`   | `treat_source_earned`      | **−1.700** (strong demoter) |
| `T5_topical_comp`    | `treat_topical_comp`       | **+0.438** (strong promoter) |
| `T3_structured_data_new` | `treat_structured_data`| −0.140 |
| `T2a_question_headings`  | `treat_question_headings`| +0.103 |
| `T6_freshness`       | `treat_freshness`          | −0.060 |
| `T1b_stats_density`  | `treat_stats_density`      | −0.017 |

Full list: `geodml_data/docs/treatment-confounder-dictionary.md` (after
download).

---

## Reproducibility

- Every script accepts `--seed` (default 42) and `--resume`.
- Checkpoints every 10 examples to `interpretability/output/checkpoint_<script>.json`.
- GPU memory logged at startup via `torch.cuda.memory_allocated()`.
- Output CSVs are idempotent on re-run (row-level dedupe on resume).
