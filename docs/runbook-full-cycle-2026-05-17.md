# Runbook — Full cycle (Mac → HF → JUWELS scifi → HF → Mac)

**Goal**: standardize the GEODML dataset under one inference regime
(bf16 full precision), run the JUWELS work under the new `scifi` project,
push every intermediate state to HuggingFace, and end with a Mac-side
analysis on the final precision-tagged data.

**Linear sequence** — each step depends on the previous one.

```
  ┌────────────────┐  ① push current state to HF (precision col backfilled)
  │      MAC       │
  └───────┬────────┘
          │
  ┌───────▼────────┐  ② JUWELS scifi setup from scratch
  │     JUWELS     │  ③ pull from HF
  │     scifi      │  ④ run the bf16 experiment (~3 days GPU)
  │                │  ⑤ standardize (backfill + audit)
  │                │  ⑥ push to HF (#2)
  │                │  ⑦ DML analysis (Stages B/C/D)
  │                │  ⑧ generate RESULTS_SUMMARY.md
  │                │  ⑨ push to HF (#3, final)
  └───────┬────────┘
          │
  ┌───────▼────────┐  ⑩ pull from HF, do final analysis locally
  │      MAC       │
  └────────────────┘
```

HF repo target throughout: **`ValerianFourel/geodml-papersize`** (matches
`.env`). All three pushes overwrite the same repo — HF's git history preserves
every revision for rollback.

---

## Step ① — Push current state to HF (on Mac)

Already executed. Confirms:

- All 41,383 historical records backfilled with `llm_backend` + `llm_precision`.
- All 4 Stage C parquets re-merged with the new columns.
- `README.md`, `PROVENANCE.md`, `CHANGELOG.md` updated to document the schema.
- `hf upload-large-folder` running in background — incremental upload of
  modified parquets + docs.

If you need to re-do or trigger again:

```bash
cd ~/Hamburg/geodml-dataset
set -a; source ~/Hamburg/GEODML_Analysis/.env; set +a
hf upload-large-folder \
  ValerianFourel/geodml-papersize . \
  --repo-type dataset --num-workers 4
```

Verify on the HF web UI: https://huggingface.co/datasets/ValerianFourel/geodml-papersize/tree/main

---

## Step ② — Set up JUWELS `scifi` from scratch

### 2.1 Login + activate scifi

```bash
ssh juwels   # or:  ssh <user>@juwels-booster.fz-juelich.de
jutil env activate -p scifi
echo "PROJECT=$PROJECT  SCRATCH=$SCRATCH"
sshare -A scifi -u $USER --format=Account,User,RawShares,NormShares
# RawShares > 0 → scifi membership is live for you. If error, file ticket at judoor.fz-juelich.de.
```

### 2.2 Clone repo + build venv (login node only)

```bash
cd $SCRATCH
git clone https://github.com/ValerianFourel/GEODML_Analysis.git
cd GEODML_Analysis

module load Stages/2024 GCC Python CUDA
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
pip install lightgbm doubleml rank_bm25 sentence-transformers textstat \
            bitsandbytes accelerate

# Sanity
.venv/bin/python -c "
import torch, transformers, accelerate
print(f'torch={torch.__version__}  transformers={transformers.__version__}')
print(f'cuda={torch.cuda.is_available()}  n_gpus={torch.cuda.device_count()}')
"
```

### 2.3 Configure `.env` (use scifi everywhere)

```bash
cd $SCRATCH/GEODML_Analysis
cp .env.example .env
$EDITOR .env
```

Set:

```ini
JUWELS_ACCOUNT=scifi
JUWELS_PROJECT=scifi
HF_TOKEN=hf_xxx_write_token              # write scope (you'll push back)
GEODML_DATA_ROOT=/p/scratch/scifi/$USER/geodml_data
LOCAL_PRECISION=full
MAX_KW=400
PRIMARY_MODEL=meta-llama/Llama-3.3-70B-Instruct
PROXY_MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_DATASET_REPO=ValerianFourel/geodml-papersize
```

Verify:

```bash
set -a; source .env; set +a
env | grep -E "JUWELS_|GEODML_DATA_ROOT|LOCAL_PRECISION|HF_DATASET_REPO"
```

---

## Step ③ — Pull dataset from HF on JUWELS

```bash
cd $SCRATCH/GEODML_Analysis
set -a; source .env; set +a

mkdir -p "$GEODML_DATA_ROOT"
.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    'ValerianFourel/geodml-papersize',
    repo_type='dataset',
    local_dir=os.path.dirname(os.environ['GEODML_DATA_ROOT']),
    allow_patterns=['data/**', 'PROVENANCE.md', 'README.md', 'CHANGELOG.md'],
)
print('snapshot OK')
"

# Sanity: precision column in the freshly-pulled parquets
.venv/bin/python -c "
import pandas as pd, os
ROOT = os.environ['GEODML_DATA_ROOT'] + '/data/main'
for v in ['biased','neutral','biased_rag','neutral_rag']:
    df = pd.read_parquet(f'{ROOT}/full_experiment_data_{v}.parquet')
    print(f'{v:14s}  precision={df[\"llm_precision\"].value_counts().to_dict()}')
"
```

### 3.5 Pre-populate HF model cache (compute nodes are offline)

```bash
export HF_HOME=$SCRATCH/hf_cache
mkdir -p "$HF_HOME"
.venv/bin/python -c "
from huggingface_hub import snapshot_download
for m in ['meta-llama/Llama-3.3-70B-Instruct',
          'Qwen/Qwen2.5-72B-Instruct',
          'meta-llama/Llama-3.1-8B-Instruct']:
    print('downloading', m); snapshot_download(m, cache_dir='$HF_HOME')
"
# 70B × 2 + 8B at bf16 ≈ 300 GB. Plan ~2 hours. Run in tmux.
```

---

## Step ④ — Run the bf16 experiment

### 4.1 Smoke first (15 min, develbooster, 5 keywords)

```bash
cd $SCRATCH/GEODML_Analysis
set -a; source .env; set +a

MAX_KW=5 LOCAL_PRECISION=full \
  ./scripts/slurm/dispatch_all.sh --smoke \
    --only rerank --variant biased \
    --models meta-llama/Llama-3.3-70B-Instruct \
    --engines searxng --pools 20
```

Watch the log:

```bash
JID=$(squeue -u $USER -h -o '%i' | head -1)
tail -f logs/rerank-*-$JID*.out
```

**Smoke pass criteria** (look for these lines):

- `[LocalRanker] model=meta-llama/Llama-3.3-70B-Instruct precision=bf16-full`
- `[LocalRanker] hf_device_map: ... 4 devices`
- `[LocalRanker]   cuda:N allocated=~35 GiB` per GPU
- `[rerank] backend=local ... precision=full`

Then verify the JSONL carries the new field:

```bash
.venv/bin/python -c "
import json
p='$GEODML_DATA_ROOT/data/runs/searxng_Llama-3.3-70B-Instruct_serp20_top10_biased/phase2/keywords.jsonl'
rec = json.loads(open(p).readline())
assert rec['llm_parameters']['precision'] == 'bf16-full', rec['llm_parameters']
print('smoke OK — precision=bf16-full confirmed')
"
```

### 4.2 Full launch (~2-3 days, scifi accounting)

```bash
MAX_KW=400 LOCAL_PRECISION=full \
  INCLUDE_RAG_REDO=1 INCLUDE_F_GAPS=1 \
  ./scripts/finish_on_gpu.sh \
    2>&1 | tee logs/finish_on_gpu_$(date +%Y%m%d_%H%M%S).log
```

What this submits:

- 16 snippet rerank jobs (biased + neutral × 2 models × 2 engines × 2 pools)
- 16 RAG rerank jobs (the `INCLUDE_RAG_REDO=1` redo at bf16)
- 64 order_probe jobs (4 variants × 2 models × 2 engines × 2 pools × 2 seeds)
- Stage F gap fills (probing for `neutral`; ablation/saliency/probing/weights
  for both `_rag` variants)

The `skip_if_at_max` pre-flight guard in each sbatch silently exits any
cell already at `MAX_KW=400` keywords — so re-launching is idempotent.

### 4.3 Monitor

```bash
watch -n 30 'squeue -u $USER --format="%.10i %.32j %.2t %.10M %.10l" | head -30'
# Or check the audit dashboard:
.venv/bin/python scripts/audit_status.py | tail -20
```

---

## Step ⑤ — Standardize everything on JUWELS

Wait until `squeue -u $USER` is empty.

```bash
cd $SCRATCH/GEODML_Analysis
set -a; source .env; set +a

# 5.1 Backfill precision into ALL JSONLs (idempotent; covers any new files).
.venv/bin/python scripts/backfill_precision.py \
  --root "$GEODML_DATA_ROOT" --include-recent --report-only
# Expect: records_total≈80k, records_already_set≈80k (everything already written
# correctly by the new code path). If patches>0, there's a code drift — investigate.

# 5.2 Confirm every cell is at MAX_KW.
.venv/bin/python scripts/audit_status.py | tail -30
# Expect: Stage A 32/32, Stage A' 96/96, Stage F 100% for all 6 variants.

# 5.3 Confirm precision column is consistently bf16-full for snippet, bf16-full
#     for RAG (since INCLUDE_RAG_REDO=1), nothing else.
.venv/bin/python -c "
import json, glob, os
root = os.environ['GEODML_DATA_ROOT']
all_prec = {}
for p in sorted(glob.glob(f'{root}/data/runs/*/phase2/keywords.jsonl')):
    with open(p) as f:
        prec = {json.loads(l)['llm_parameters']['precision']
                for l in f if l.strip()}
    cell = p.split('/')[-3]
    all_prec[cell] = prec
unique = set().union(*all_prec.values())
print('Distinct precision labels across all rerank cells:', unique)
assert unique == {'bf16-full'}, f'NOT standardized: {unique}'
print('STANDARDIZED — all rerank records are bf16-full.')
"
```

---

## Step ⑥ — Push to HF (#2 — post-cluster, pre-DML)

```bash
cd $SCRATCH/GEODML_Analysis
set -a; source .env; set +a

# Update PROVENANCE on cluster-side so the push documents the new state.
# Edit by hand or just append a date-stamped entry:
cat >> "$GEODML_DATA_ROOT/CHANGELOG.md" <<EOF

## $(date -u +%Y-%m-%d) — bf16 reconciliation complete

Re-ran snippet + RAG rerank and order_probe in LOCAL_PRECISION=full
(bf16) on JUWELS scifi project. All Stage A records now \`bf16-full\`.
Stage F gaps closed (probing for neutral; full F for _rag variants).

EOF

# Push everything under $GEODML_DATA_ROOT/data (skip Stage F output for now —
# it lives outside the data dir and is bulky).
hf upload-large-folder ValerianFourel/geodml-papersize \
  "$GEODML_DATA_ROOT" \
  --repo-type dataset \
  --num-workers 4 \
  --path-in-repo data
```

---

## Step ⑦ — DML analysis on JUWELS

```bash
cd $SCRATCH/GEODML_Analysis
set -a; source .env; set +a

# Runs Stage B (variant-agnostic, cached), Stage C (merge × 4 variants),
# Stage D (DML × 4 variants), make_figures.py, order_probe_analyze.py.
bash scripts/continue_pipeline.sh
```

This regenerates:

- `data/main/full_experiment_data_{biased,neutral,biased_rag,neutral_rag}.parquet`
  — Stage C with the new `llm_precision` column.
- `data/dml_results/dml_results_long_{biased,neutral,biased_rag,neutral_rag}.parquet`
  — Stage D, 280 fits each across the full PLR × {LGBM, RF} × 7 subsets ×
  10 treatments × 2 outcomes grid.
- `interpretability/output/plots/figure_*.png` — refreshed figures.
- `data/order_probe/order_probe_summary.parquet` — Jaccard / OAK per cell.

Then re-run audit to confirm 100%:

```bash
.venv/bin/python scripts/audit_status.py | tail -5
.venv/bin/python scripts/audit_pipeline.py 2>&1 | tail -10
```

### 7.5 Headline sanity check

```bash
.venv/bin/python -c "
import pandas as pd, os
root = os.environ['GEODML_DATA_ROOT']
for v in ['biased','neutral','biased_rag','neutral_rag']:
    df = pd.read_parquet(f'{root}/data/dml_results/dml_results_long_{v}.parquet')
    r = df.query('subset==\"POOLED\" and method==\"plr\" and learner==\"lgbm\" '
                 'and outcome==\"rank_delta\" and treatment==\"T7_source_earned\"').iloc[0]
    print(f'{v:14s}  T7_source_earned: {r.coef:+.3f}  p={r.p_val:.4f}')
"
# Pre-bf16 reference (from data/dml_results/dml_results_long_ALL.csv):
#   biased       T7 = -1.607 ***
#   neutral      T7 = -0.417 ***
#   biased_rag   T7 = -1.268 ***
#   neutral_rag  T7 = -0.496 ***
# Post-bf16: expect each within ±15% of these, same sign, still p<0.001.
# If wildly different, document — that IS a paper-worthy result.
```

---

## Step ⑧ — Generate `RESULTS_SUMMARY.md`

```bash
cd $SCRATCH/GEODML_Analysis
.venv/bin/python scripts/make_results_summary.py \
  --data-root "$GEODML_DATA_ROOT" \
  --variants biased neutral biased_rag neutral_rag \
  --output "$GEODML_DATA_ROOT/RESULTS_SUMMARY.md" \
  --title "GEODML — Final results (bf16 reconciliation, $(date -u +%Y-%m-%d))"

# Spot-check
head -60 "$GEODML_DATA_ROOT/RESULTS_SUMMARY.md"
```

This produces a single markdown that covers:

1. Per-variant inventory (rows, fits, precision breakdown).
2. Headline coefficient table on POOLED · plr · lgbm · rank_delta.
3. The 2×2 grid (prompt × method) for T7 and T5 + interaction terms.
4. Per-subset T7 breakdown.
5. Precision stratification per variant.
6. Companion analysis files (rag_vs_snippet_comparison.csv, etc.).
7. Reproducibility — exact regeneration command.

---

## Step ⑨ — Push to HF (#3 — final, with DML results + summary)

```bash
cd $SCRATCH/GEODML_Analysis
set -a; source .env; set +a

# This time push the FULL tree: data + Stage F output + RESULTS_SUMMARY.md
# (Build a transient consolidated dir so html_caches aren't re-uploaded — they
# haven't changed since step ⑥.)
hf upload-large-folder ValerianFourel/geodml-papersize \
  "$GEODML_DATA_ROOT" \
  --repo-type dataset \
  --num-workers 4

# Also push the Stage F outputs as a separate sub-tree.
hf upload-large-folder ValerianFourel/geodml-papersize \
  "$SCRATCH/GEODML_Analysis/interpretability/output" \
  --repo-type dataset \
  --num-workers 4 \
  --path-in-repo interpretability/output
```

Confirm on HF web UI that:

- `RESULTS_SUMMARY.md` is at the repo root.
- `data/dml_results/dml_results_long_*.parquet` files have a recent timestamp.
- `data/main/full_experiment_data_*.parquet` shows `llm_precision` =
  `bf16-full` across both snippet AND RAG variants.

---

## Step ⑩ — Pull on Mac, do final analysis

```bash
cd ~/Hamburg/GEODML_Analysis
set -a; source .env; set +a

# 10.1 Pull the lean analysis subset (skip html_caches — 40 GB you don't
# need for analysis).
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'ValerianFourel/geodml-papersize',
    repo_type='dataset',
    local_dir='$HOME/Hamburg/geodml-dataset',
    allow_patterns=[
        'data/main/**', 'data/dml_results/**', 'data/features/**',
        'data/order_probe/order_probe_summary.parquet',
        'interpretability/output/**',
        'RESULTS_SUMMARY.md', 'README.md', 'PROVENANCE.md', 'CHANGELOG.md',
    ],
)
print('lean snapshot OK')
"

# 10.2 Open the auto-generated summary.
$EDITOR ~/Hamburg/geodml-dataset/RESULTS_SUMMARY.md

# 10.3 Load DML results for ad-hoc analysis.
.venv/bin/python -c "
import pandas as pd
ROOT = '$HOME/Hamburg/geodml-dataset/data/dml_results'
dfs = {v: pd.read_parquet(f'{ROOT}/dml_results_long_{v}.parquet')
       for v in ['biased','neutral','biased_rag','neutral_rag']}

# Build the 2×2 grid for T7 with confidence intervals.
import functools
def headline(df, t='T7_source_earned'):
    return df.query(
        f'subset==\"POOLED\" and method==\"plr\" and learner==\"lgbm\" '
        f'and outcome==\"rank_delta\" and treatment==\"{t}\"'
    ).iloc[0]

for v, df in dfs.items():
    r = headline(df)
    print(f'{v:14s}  T7  coef={r.coef:+.3f}  CI=[{r.ci_lower:+.3f},{r.ci_upper:+.3f}]  '
          f'p={r.p_val:.4f}  n={r.n_obs:,}')

# Now compare snippet vs RAG within each prompt
print()
print('Snippet → RAG shift per prompt:')
for prompt in ['biased', 'neutral']:
    s = headline(dfs[prompt])
    r = headline(dfs[f'{prompt}_rag'])
    shift = r.coef - s.coef
    print(f'  {prompt:8s}: {s.coef:+.3f} (snippet) → {r.coef:+.3f} (RAG)  '
          f'Δ={shift:+.3f}')
"

# 10.4 Generate figures, presentations, paper drafts — your usual flow.
# The auto-summary feeds presentation-ready tables for any markdown deck.
```

---

## Failure modes + recovery

### A. JUWELS scifi project not yet active
- `jutil env activate -p scifi` errors with "no such project".
- Fix: ticket at https://judoor.fz-juelich.de — don't proceed.

### B. Smoke test (step 4.1) shows `precision=4bit-nf4`
- Cause: `LOCAL_PRECISION` not exported by sbatch.
- Fix: re-submit with `--export=ALL,LOCAL_PRECISION=full,...`. Discard the
  contaminated cell:
  ```bash
  V=biased; CELL=searxng_Llama-3.3-70B-Instruct_serp20_top10_${V}
  mv $GEODML_DATA_ROOT/data/runs/$CELL/phase2/keywords.jsonl \
     $GEODML_DATA_ROOT/data/runs/$CELL/phase2/keywords.jsonl.bak
  rm $GEODML_DATA_ROOT/data/runs/$CELL/phase2/.rerank_ckpt.json
  ```

### C. OOM on Qwen-72B bf16 (step 4.2)
- Cause: 72B bf16 + RAG 2400-char prompts at edge of 4×80 GB.
- Fix:
  ```bash
  sed -i 's/--top-k-rag 3/--top-k-rag 2/' scripts/slurm/run_rerank.sbatch
  ```

### D. HF push interrupted (step 1, 6, or 9)
- `hf upload-large-folder` is resumable. Just re-run the same command —
  it skips files already uploaded with matching checksum.

### E. DML coef wildly different from pre-bf16 reference (step 7.5)
- E.g. T7 jumps from −1.61 to −0.5 or flips sign.
- Don't push (step 9) until you understand why. Likely candidates:
  - precision regime correctly differs (paper-worthy finding — document it
    in CHANGELOG before pushing).
  - Stage B features parquet was stale (re-run `interpretability.pipeline.features`).
  - LLM model version changed (check `meta.json` in rag_index).
- If confirmed, push but add a `RESULTS_SUMMARY.md` § "Notable deltas from
  4-bit baseline".

### F. Mac pull (step 10) shows missing parquets
- Cause: HF push (step 9) didn't propagate yet — HF takes ~1-2 min to
  index large updates.
- Fix: wait, then re-run the snapshot_download (idempotent).

---

## Budget

| Step | Wall time | GPU-hr (scifi) | $$ |
|---|---|---:|---|
| ① Mac push to HF | ~2-4 hr (mostly upload of new html_caches) | 0 | 0 |
| ② JUWELS setup | ~30 min | 0 | 0 |
| ③ Pull + cache models | ~2 hr | 0 | 0 |
| ④ Full bf16 experiment | ~3 days (with INCLUDE_RAG_REDO + F_GAPS) | ~230 | 0 |
| ⑤ Standardize | ~10 min | 0 | 0 |
| ⑥ Push #2 | ~30 min | 0 | 0 |
| ⑦ DML on login node | ~30 min | 0 | 0 |
| ⑧ Generate summary | 1 min | 0 | 0 |
| ⑨ Push #3 | ~10 min | 0 | 0 |
| ⑩ Mac pull + analysis | ~10 min | 0 | 0 |

End-to-end wall time: ~4-5 days. scifi GPU consumption: ~230 hours.

---

## Quick-reference command index

```bash
# Mac → HF (this push, already running)
cd ~/Hamburg/geodml-dataset && hf upload-large-folder \
  ValerianFourel/geodml-papersize . --repo-type dataset --num-workers 4

# JUWELS pull
.venv/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('ValerianFourel/geodml-papersize', repo_type='dataset', \
  local_dir='$SCRATCH/geodml_data_parent')"

# JUWELS run
MAX_KW=400 LOCAL_PRECISION=full INCLUDE_RAG_REDO=1 INCLUDE_F_GAPS=1 \
  ./scripts/finish_on_gpu.sh

# JUWELS standardize + push
.venv/bin/python scripts/backfill_precision.py --root $GEODML_DATA_ROOT --include-recent
hf upload-large-folder ValerianFourel/geodml-papersize $GEODML_DATA_ROOT \
  --repo-type dataset --num-workers 4 --path-in-repo data

# JUWELS DML
bash scripts/continue_pipeline.sh

# JUWELS summary + final push
.venv/bin/python scripts/make_results_summary.py --data-root $GEODML_DATA_ROOT
hf upload-large-folder ValerianFourel/geodml-papersize $GEODML_DATA_ROOT \
  --repo-type dataset --num-workers 4

# Mac pull (analysis subset only)
.venv/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('ValerianFourel/geodml-papersize', repo_type='dataset', \
  local_dir='$HOME/Hamburg/geodml-dataset', \
  allow_patterns=['data/main/**','data/dml_results/**','RESULTS_SUMMARY.md'])"
```

End.
