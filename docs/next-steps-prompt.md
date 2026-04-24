# Prompt: Run the GEODML Interpretability Pipeline on a GPU box

You are taking over a partially finished **mechanistic-interpretability**
pipeline for an **EMNLP 2026** submission. **Paper deadline: 2026-05-25.**

All code is ready. Your job is to execute it on a GPU machine (Jülich HPC
or similar), verify each stage, and hand back three paper-ready figures.

---

## 1. What this is

The parent paper is a Double Machine Learning (DML) study of how an LLM
re-ranks web-search results. It identifies 15 page-level content
treatments with causal effects on rank (top hits: **T7_source_earned**
strong demoter, coef = −1.700; **T5_topical_comp** strong promoter, coef
= +0.438).

This pipeline adds three orthogonal mech-interp analyses that validate
and extend those estimates:

| Option | Method | Needs | Runtime |
|---|---|---|---|
| 1. **Ablation** | Remove a treatment-relevant feature from each SERP candidate's snippet, re-rank, measure rank delta. | Any re-ranker backend. | 2–4 h |
| 2. **Saliency** | Gradient × embedding saliency on a local 4-bit proxy model; tag tokens by treatment and compute saliency ratios. | 1× GPU, ≥12 GB VRAM. | 1–2 h |
| 3. **Probing** | Train a logistic probe per transformer layer to decode the treatment label from hidden states. | 1× GPU, ≥10 GB VRAM. | 30–60 min |

Outputs → Figure A (ablation vs DML scatter), Figure B (saliency
heatmap), Figure C (probing curves).

---

## 2. Where everything lives

**Code:** `https://github.com/ValerianFourel/GEODML_Analysis` (private).

**Data:** HF dataset `ValerianFourel/geodml-papersize` (private). ~3.5 GB
without HTML caches; ~28 GB after extraction. Everything you need is
there — the SERP snapshots (`data/serp/phase0_*.parquet`) and the
per-page HTML (`data/runs/*/phase2/html_cache.tar.gz`) are cached. You
do **not** need SearXNG, DuckDuckGo, or any live HTTP scrape.

**Tokens:** `HF_TOKEN` (read on the dataset) and `GH_TOKEN` (pull the
repo) must go in `.env`. `.env` is gitignored — never commit it.

---

## 3. Air-gapped clusters (Jülich-specific)

Compute nodes on Jülich have no outbound HTTP. Login nodes usually do.
Stage everything on the login node, then run on the compute node.

```bash
# ---- login node ----
git clone https://github.com/ValerianFourel/GEODML_Analysis.git
cd GEODML_Analysis
cp .env.example .env        # then fill in HF_TOKEN
python scripts/download_data.py                 # ~3.5 GB, uses HF_TOKEN
python scripts/download_data.py --no-download --extract-html   # +24 GB
# Copy to fast scratch if needed:
# rsync -a geodml_data/ "$SCRATCH/geodml_data/"

# ---- compute node, GPU partition ----
module load Python CUDA      # or similar, depending on the site
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install "torch>=2.4" "transformers>=4.45" "accelerate>=0.34" "bitsandbytes>=0.43"

# For ablation, use the offline backend (no HF Inference API calls):
ABLATION_BACKEND=local ./run_all.sh
```

If the compute node can't see the Hub either, set
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` after pre-caching the
proxy model to `$HF_HOME`.

---

## 4. Execution order

Follow this order. Each step is resumable via `--resume`; checkpoints
land in `interpretability/output/checkpoint_*.json`.

### Step 0 — verify the DML table

```bash
python scripts/sanity_check.py
```

Expected: 15 significant POOLED rows on `rank_delta`.
**T7_source_earned** coef ≈ −1.700, **T5_topical_comp** coef ≈ +0.438.
If those two signs are wrong, stop and investigate — something is off
with the data layout.

### Step 1 — Ablation (Option 1)

```bash
python -m interpretability.ablation --sample-n 500 --resume \
    --backend local --models meta-llama/Llama-3.1-8B-Instruct
# or if the cluster has ≥40 GB VRAM and can hold a 70B in 4-bit:
#   --models meta-llama/Llama-3.3-70B-Instruct
```

**Output:** `interpretability/output/ablation_results.csv` with columns
`keyword, url, domain, treatment, model, baseline_rank, ablated_rank,
ablation_delta`.

**Sanity:** for T7_source_earned, mean `ablation_delta` should be
*positive* (ablating the earned-media demoter **promotes** the page, so
baseline_rank > ablated_rank). Sign should flip for T5_topical_comp.

**Model-size caveat:** an 8B proxy will not match the 70B/72B DML
coefficients in magnitude. Sign and relative ordering should still hold.
Report this in the paper.

### Step 2 — Saliency (Option 2)

```bash
python -m interpretability.saliency --sample-n 200 --resume
```

Defaults to `$LOCAL_MODEL` (Llama-3.1-8B-Instruct). Balanced 100/100
earned vs brand keywords.

**Output:**
- `saliency_scores.csv` — per-token saliency, tagged by treatment.
- `saliency_summary.csv` — per-treatment mean saliency ratio (treatment
  tokens / other tokens).

**Sanity:** for T7_source_earned, `saliency_ratio > 1` means the model
does attend to URL/domain tokens — which is the mechanistic evidence
that backs the large DML coefficient.

### Step 3 — Probing (Option 3)

```bash
python -m interpretability.probing --sample-n 2000 --resume \
    --run-filter searxng_Qwen2.5-72B-Instruct_serp50_top10
```

Trains 32 × 4 logistic probes (one per layer, per treatment in
{T7, T5, T2a, T6}), two pooling variants (`last_token`, `mean`).

**Output:** `probing_results.csv` — `treatment, layer, pooling,
accuracy, roc_auc, n_train, n_test`.

**Hypothesis to check:** T7 (source/surface signal) decodable from
early-mid layers; T5 (topical competence) peaks later. The probing
curves in Figure C should reflect this.

### Step 4 — Figures

```bash
python -m interpretability.make_figures
```

Produces three PNGs in `interpretability/output/plots/`:

- `figure_a_ablation.png` — DML coef (x) vs mean ablation_delta (y),
  with the y = −x reference line.
- `figure_b_saliency.png` — heatmap of mean saliency per treatment.
- `figure_c_probing.png` — probe accuracy by layer, per treatment, two
  panels (last_token and mean pooling).

Drop straight into the paper's `figures/` directory.

---

## 5. Treatment ↔ column cheat sheet

DML reports coefficients per **treatment label**. The main-table parquet
uses **different column names**. Do not confuse them.

| DML label | main-table column | type | direction on rank_delta |
|---|---|---|---|
| T7_source_earned | `treat_source_earned` | 0/1 | **−1.700** (demoter) |
| T5_topical_comp | `treat_topical_comp` | float | **+0.438** (promoter) |
| T3_structured_data_new | `treat_structured_data` | 0/1 | −0.140 |
| T6_freshness | `treat_freshness` | 0–4 | −0.060 |
| T2a_question_headings | `treat_question_headings` | 0/1 | +0.103 |
| T1b_stats_density | `treat_stats_density` | float | −0.017 |

Full map: `interpretability/utils.TREATMENT_TO_COL`.
Full dictionary: `geodml_data/docs/treatment-confounder-dictionary.md`.

---

## 6. Rules of engagement

- **Do not commit `.env`** or any file containing tokens. `.gitignore`
  already excludes it; verify with `git check-ignore .env` before every
  commit.
- **Do not re-download** the dataset if `geodml_data/` is populated —
  `snapshot_download` is idempotent, but rerunning costs bandwidth you
  don't need.
- **Do not tweak the re-ranking prompt** in `utils.build_rerank_prompt`.
  It is transcribed verbatim from the original `src/llm_ranker.py` —
  changing it invalidates the comparison with DML.
- **Commit outputs only if small** (CSVs < 10 MB, plots). Large
  checkpoints stay local.
- **Push** after each analysis completes:

  ```bash
  git add interpretability/output/*.csv interpretability/output/plots/*.png
  git commit -m "results: <option>"
  git push
  ```

---

## 7. When you are done

1. `interpretability/output/` has the three CSVs and three PNGs.
2. `git status` is clean.
3. Post a short summary: for each of the three options, one sentence on
   whether the direction of effect matched the DML signs and one
   sentence on the headline number (e.g. "T7 saliency ratio = 3.2" or
   "T5 probe peaks at layer 24").

That summary is what goes into Section 5 (Interpretability) of the
paper. The figures are the evidence.

---

## 8. If you get stuck

- OOM on saliency/probing → lower `--batch-size` (probing) or shorten
  `--max-len` (probing, default 512).
- API rate limits on ablation (if using `--backend api`) → switch to
  `--backend local`.
- HTML cache miss → check `geodml_data/data/runs/{run_id}/phase2/` has
  both `html_cache.tar.gz` *and* an extracted `html_cache/` dir. The
  extractor is idempotent; re-run `scripts/download_data.py
  --no-download --extract-html` if in doubt.
- Probe accuracy flat at 0.5 across all layers → the treatment column
  may have no variance in the sampled run. Try `--run-filter` a
  different run_id, or drop the filter to use all 8 runs.

Everything else: read `interpretability/utils.py` first — it's the
single source of truth for data loading, prompt building, and ranker
backends.
