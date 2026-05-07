# Continue the pipeline locally (CPU-only)

The cluster snapshot you uploaded contains all the expensive LLM work
(rerank, order probe) and most of the interpretability outputs (ablation,
saliency, weights, 2/8 probing). What's left to compute is **deterministic
and CPU-only**, so you can finish it on a laptop or a small vast.ai box.

## The plan

| Stage | Status now | What it needs | Where it runs |
|---|---|---|---|
| A — rerank | 24/32 | done | — |
| A' — order probe + summary | 48/64 + summary parquet | done | — |
| B — features | **0/4** | cached HTML + regex/BS4 | **CPU, this guide** |
| C — merge main table | **0/4** | Stage B output, pandas | **CPU, this guide** |
| D — DML | **0/4** | Stage C output, sklearn | **CPU, this guide** |
| F — ablation | 48/48 | done | — |
| F — saliency | 16/16 | done | — |
| F — weights | 8/8 | done | — |
| F — probing | 2/8 | hidden states from 70B → **GPU only** | JUWELS |

The 8 missing Stage A cells (ddg × passage variants) and 16 missing Stage A'
cells **cannot be filled by API** — they need ddg HTML, which doesn't exist
anywhere. The 6 missing probing cells need raw model hidden states, which no
inference API exposes. Leave those.

## What you'll get out

After running this guide on a 4-core laptop (~30–60 min wall time):

- `data/features/features_searxng_top{20,50}.parquet` — Stage B (2/4)
- `data/main/full_experiment_data_{variant}.parquet` × 4 — Stage C (4/4)
- `data/dml_results/dml_results_long_{variant}.parquet` × 4 — Stage D (4/4)
- `interpretability/output/plots/figure_{a,b,c}_*.png` — paper figures

Stage B will only complete for searxng (the only engine with HTML cache).
ddg cells get gracefully skipped end-to-end. The DML headline numbers come
from POOLED rows on searxng anyway, so this is the result table you need.

## Setup — laptop

```bash
# 1. Clone & venv
git clone https://github.com/<you>/GEODML_Analysis.git
cd GEODML_Analysis
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 2. HF token (read-only is fine for downloads)
cp .env.example .env
$EDITOR .env   # paste HF_TOKEN

# 3. Pull the snapshot. Two paths:

# 3a. Light path (~3 GB transfer): the two zip archives you uploaded,
#     plus html_cache only for searxng (the only engine that has it).
mkdir -p archives
hf download ValerianFourel/geodml-papersize \
  archives/geodml_data_.zip archives/interpretability_.zip \
  --repo-type dataset --local-dir .
unzip -q archives/geodml_data_.zip
unzip -q archives/interpretability_.zip
# Pull missing html_cache from HF (sync_data.py only fetches files that
# aren't already on disk — should grab the 5 html_cache.tar.gz files):
python scripts/sync_data.py
export GEODML_DATA_ROOT="$PWD/geodml_data"

# 3b. Full path (~28 GB): mirror the entire HF dataset.
python scripts/download_data.py --extract-html
export GEODML_DATA_ROOT="$PWD/geodml_data"
```

## Setup — vast.ai (CPU instance)

Pick the cheapest CPU instance (no GPU needed). 4 vCPU + 8 GB RAM is enough.

```bash
# inside the vast.ai container
apt-get update && apt-get install -y python3 python3-venv unzip git
git clone https://github.com/<you>/GEODML_Analysis.git
cd GEODML_Analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt

# get the snapshot (use the light path above)
echo "HF_TOKEN=hf_xxx_read_token_here" > .env
mkdir -p archives
hf download ValerianFourel/geodml-papersize \
  archives/geodml_data_.zip archives/interpretability_.zip \
  --repo-type dataset --local-dir .
unzip -q archives/geodml_data_.zip
unzip -q archives/interpretability_.zip
python scripts/sync_data.py     # adds the html_cache tarballs

export GEODML_DATA_ROOT="$PWD/geodml_data"
```

Expected disk footprint: ~5 GB after Stage B/C/D run.

## Run

```bash
bash scripts/continue_pipeline.sh
```

Or smoke-test with a small keyword cap first:

```bash
FEATURES_MAX_KW=50 bash scripts/continue_pipeline.sh
# verify Stage D produced reasonable rows, then re-run without the cap
bash scripts/continue_pipeline.sh
```

The script is **idempotent** — re-running skips any (engine, pool) or
(variant) that already has a non-empty output parquet. To force recompute:

```bash
FORCE=1 bash scripts/continue_pipeline.sh
```

## After the run — packaging & upload

The script's last steps automatically:

1. **Zip the new artifacts** into
   `archives/local_results_<date>-<hhmm>.zip`. The zip contains:
   - `geodml_data/data/features/` — Stage B parquets
   - `geodml_data/data/main/` — Stage C merged tables
   - `geodml_data/data/dml_results/` — Stage D long tables
   - `interpretability/output/plots/` — regenerated paper figures
   - `PROVENANCE.txt` — git commit, host, date, and the LLM config in
     effect (`temperature=0.1`, `max_tokens=500`) so future readers can
     verify these results were generated with the same settings as the
     cluster runs.

2. **Print the upload command** ready to copy/paste — e.g.:
   ```bash
   huggingface-cli login
   hf upload ValerianFourel/geodml-papersize \
     archives/local_results_20260507-1530.zip \
     archives/local_results_20260507-1530.zip \
     --repo-type dataset \
     --commit-message "local Stage B/C/D 20260507-1530"
   ```
   Single-file commit → no 500 errors regardless of size.

The script does **not** auto-upload — you review the zip first, then run
the printed command with a write-scoped token.

## Stage F is a separate step (not in this script)

Stage F is 74/80 already; the 6 missing cells are all probing
(neutral, biased_passage, neutral_passage × {Llama, Qwen}). Probing
needs raw 70B-model hidden states (`output_hidden_states=True`), which no
inference API exposes — it has to run on a GPU box with the weights
loaded.

**On JUWELS:**
```bash
# 6 cells = 2 models × 3 variants. ~4–10 h wall on 4×A100-40G each.
for MODEL in meta-llama/Llama-3.3-70B-Instruct Qwen/Qwen2.5-72B-Instruct; do
  for V in neutral biased_passage neutral_passage; do
    sbatch --account=$JUWELS_ACCOUNT \
      --export=ALL,MODEL=$MODEL,PROMPT_VARIANT=$V \
      scripts/slurm/run_probing.sbatch
  done
done
```

Once those land, re-run `audit_pipeline.py` on JUWELS to confirm
Stage F is 80/80, then re-package and upload (same script, just from
JUWELS this time). The local Stage B/C/D results from this guide are
unaffected by Stage F finishing.

## What you should see at the end

The audit at the bottom of the local run should show:

```
Stage A   24/32   Stage B   2/4    Stage C   4/4    Stage D   4/4    Stage F  74/80   Order probe  48/64
```

Specifically:
- **Stage B 2/4** — searxng × {20,50} done; ddg × {20,50} permanently
  skipped (no HTML).
- **Stage C 4/4** — all 4 variants merged. Each contains only searxng cells
  for the 2 *_passage variants; biased / neutral additionally include ddg
  rerank rows (with NaN treatments since ddg has no features).
- **Stage D 4/4** — DML grid fitted on each variant. Look at:
  ```
  Headline — POOLED, plr, lgbm, rank_delta  (Δ = neutral − biased)
  ```
  in the audit — that's your headline result.

If `make_figures` succeeded, paper figures are at:
```
interpretability/output/plots/figure_a_ablation_{full,rw}.png
interpretability/output/plots/figure_b_saliency_{full,rw}.png
interpretability/output/plots/figure_c_probing.png
```

## Why no API step?

The remaining LLM work is gated on inputs we don't have:

- **Rerank ddg × passage**: needs ~800 chars of cleaned body text per
  result, sourced from the cached HTML (`prompts.py` `_PASSAGE_MAX_CHARS`).
  No ddg HTML was ever cached → no passages to feed an API.
- **Order probe ddg × passage**: same prompt, same blocker.
- **Probing**: needs `output_hidden_states=True` on the 70B model.
  Inference APIs return logprobs / output text, never per-layer activations.

Re-scraping ddg URLs would change the experiment (URLs and content drift),
so the right call is to leave those cells unfilled and report results on
searxng. The DML POOLED headline includes both engines for snippet variants
(biased/neutral) and is searxng-only for passage variants — make this
explicit in the paper.

## Troubleshooting

**"no html_cache for engine=searxng pool=20"** — `sync_data.py` didn't pull
the tarballs, or you unzipped into a different directory than
`$GEODML_DATA_ROOT`. Verify:
```bash
ls "$GEODML_DATA_ROOT"/data/runs/searxng_Llama-3.3-70B-Instruct_serp20_top10/phase2/html_cache*
```

**features.py is slow** — the `all-MiniLM-L6-v2` embedder loads on first
run (~80 MB download). After that it's CPU-bound on roughly 2k urls/min.
Skip it entirely with `--no-embed` if you don't need T5/cosine confounders:
```bash
python -m interpretability.pipeline.features --engine searxng --pool 50 --no-embed --resume
```

**DML errors on a variant** — usually means too few rows survived the
treatment + outcome dropna. Check:
```bash
python -c "
import pandas as pd
df = pd.read_parquet('$GEODML_DATA_ROOT/data/main/full_experiment_data_biased.parquet')
print(df.shape, df['rank_delta'].notna().sum())
print(df['search_engine'].value_counts())
"
```
