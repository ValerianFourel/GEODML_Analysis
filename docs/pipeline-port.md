# GEODML pipeline port — operator runbook

This repo is now self-contained: it produces DML coefficients from cached SERPs
and HTML, then runs the interpretability follow-ups (ablation, saliency,
probing, weight analysis) on top. End-to-end runs on JUWELS with sbatch.

The original (biased) prompt and a new (neutral) prompt coexist via a
``PROMPT_VARIANT`` flag. Outputs are suffixed ``_biased`` / ``_neutral`` so
the two runs do not clobber each other.

---

## 1. The pipeline at a glance

| Stage | Module | Compute | Walltime | What |
|---|---|---|---|---|
| A | ``interpretability.pipeline.rerank`` | GPU (booster) | ~2 h / job | LLM rerank cached SERPs (one job per model × engine × pool × variant) |
| B | ``interpretability.pipeline.features`` | CPU (or 1 GPU for embeddings) | ~6 h / 30 min | Deterministic T1–T7 + confounders from cached HTML |
| C | ``interpretability.pipeline.merge`` | Login / CPU | ~30 s | Join rerank logs + features into `full_experiment_data_{variant}.parquet` |
| D | ``interpretability.pipeline.dml`` | CPU (batch) | ~50 min | DoubleML PLR/IRM grid → `dml_results_long_{variant}.parquet` |
| F | ``interpretability.{ablation,saliency,probing,weight_analysis}`` | GPU (booster) | hours | Existing interp runs, now PROMPT_VARIANT-aware |
| -- | ``interpretability.make_figures`` | CPU | seconds | Paper figures, side-by-side biased + neutral |

No live HTTP from compute nodes. SERPs and HTML come from the cached HF
dataset bundle staged on `$SCRATCH/geodml_data` (set via `GEODML_DATA_ROOT`).

---

## 2. One-time setup on the login node

```bash
cd /p/project1/obdifflearn/vfourel/GEODML_Analysis     # or wherever your clone is
git pull

# Activate the venv that lives off $HOME (use a project-space venv; see
# README for the rationale — JUWELS $HOME has tight inode quotas).
source ~/.venv/bin/activate    # or: source /p/project1/obdifflearn/vfourel/venvs/geodml/bin/activate

# Install ports' new deps if they aren't there yet.
uv pip install textstat rank-bm25 doubleml lightgbm sentence-transformers

# Pre-cache sentence-transformers/all-MiniLM-L6-v2 on the login node so the
# compute nodes (HF_HUB_OFFLINE=1) can find it.
HF_HOME="$SCRATCH/hf_cache" python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2')
print('cached')
"

# Set required env in .env (read by every sbatch wrapper via _common.sh):
#   JUWELS_ACCOUNT=obdifflearn
#   JUWELS_PROJECT=obdifflearn
#   HF_TOKEN=...
```

---

## 3. Smoke test (50 keywords, 1 model, 1 engine)

Before submitting full chains, prove every stage works end-to-end on a tiny
slice. Run from a `salloc` interactive booster session (so you're on a real
GPU node):

```bash
salloc --account=obdifflearn --partition=develbooster --nodes=1 --gres=gpu:4 --time=01:30:00
srun --nodes=1 -c 10 --cpu-bind=cores --pty /bin/bash -i

cd /p/project1/obdifflearn/vfourel/GEODML_Analysis
source ~/.venv/bin/activate
export PROMPT_VARIANT=biased
export GEODML_DATA_ROOT="$SCRATCH/geodml_data"

# A: rerank 50 keywords
python -m interpretability.pipeline.rerank \
  --engine searxng --pool 50 \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --backend local --variant biased \
  --resume --max-keywords 50

# B: features for the same engine+pool (uses cached HTML cache)
python -m interpretability.pipeline.features \
  --engine searxng --pool 50 --resume

# C: merge (just this one run)
python scripts/build_main_table.py --variant biased \
  --runs searxng_Llama-3.3-70B-Instruct_serp50_top10_biased

# D: DML on the smoke parquet (only T7+T1b to keep it fast)
python -m interpretability.pipeline.dml --variant biased \
  --treatments T7_source_earned T1b_stats_density \
  --outcomes rank_delta \
  --learners lgbm --methods plr --subsets POOLED \
  --resume

# Inspect:
python - <<'PY'
import pandas as pd, os
p = f"{os.environ['GEODML_DATA_ROOT']}/data/dml_results/dml_results_long_biased.parquet"
df = pd.read_parquet(p)
print(df.to_string())
PY
```

If all four print sensible output and the DML coefficients are non-NaN,
proceed to full runs.

---

## 4. Full cluster run — biased reproduction first

The biased run is the **sanity gate** for the port. The new pipeline must
reproduce the existing biased DML coefficients to within ≈0.01 abs deviation
before you run the neutral variant.

```bash
cd /p/project1/obdifflearn/vfourel/GEODML_Analysis

# A: rerank (8 jobs: 2 models × 2 engines × 2 pools × variant=biased)
./scripts/slurm/dispatch_all.sh --only rerank --variant biased

# B: features (4 jobs: 2 engines × 2 pools, variant-agnostic)
./scripts/slurm/dispatch_all.sh --only features

# C+D: merge + DML (1 job; merge runs inline at job start)
./scripts/slurm/dispatch_all.sh --only dml --variant biased

# Verify reproduction (Day 2.0 hard gate):
python -c "
import pandas as pd, os
new = pd.read_parquet(f'{os.environ[\"GEODML_DATA_ROOT\"]}/data/dml_results/dml_results_long_biased.parquet')
old = pd.read_parquet(f'{os.environ[\"GEODML_DATA_ROOT\"]}/data/dml_results/dml_results_long.parquet')
keys = ['subset','outcome','treatment','method','learner']
m = new.merge(old, on=keys, suffixes=('_new','_old'))
diffs = (m['coef_new'] - m['coef_old']).abs()
print(f'pairs={len(m)}  median |Δcoef|={diffs.median():.4f}  max={diffs.max():.4f}')
"
```

If max ``|Δcoef| ≤ 0.01`` you are clear to proceed. If not, a port bug —
do **not** rerun with neutral until biased reproduction is solid.

---

## 5. The neutral re-run

```bash
# Same chain, variant=neutral. Outputs go to fresh _neutral-suffixed paths.
./scripts/slurm/dispatch_all.sh --only rerank --variant neutral
./scripts/slurm/dispatch_all.sh --only features              # no-op if already done
./scripts/slurm/dispatch_all.sh --only dml --variant neutral

# Re-run interpretability on the neutral DML outputs.
./scripts/slurm/dispatch_all.sh --only ablation --variant neutral
./scripts/slurm/dispatch_all.sh --only saliency --variant neutral
./scripts/slurm/dispatch_all.sh --only probing  --variant neutral

# Final figures — emits side-by-side biased+neutral plots.
python -m interpretability.make_figures
python scripts/audit_status.py
```

### One-shot orchestrator

If you would rather submit a single job that chains everything with
``--dependency=afterok``:

```bash
sbatch --account=obdifflearn \
  --export=ALL,PROMPT_VARIANT=neutral \
  scripts/slurm/run_pipeline.sbatch
```

The orchestrator submits all jobs at once and SLURM enforces ordering.
Useful if you want to walk away. Use ``SKIP_STAGES="features dml"`` to start
from rerank and let downstream stages rerun against existing parquets.

---

## 6. Variant mechanism

The active prompt variant is read at module import time of
``interpretability.pipeline.prompts``:

- ``PROMPT_VARIANT`` env var (default: ``biased``)
- Per-call ``variant=`` kwarg overrides the env

Every sbatch wrapper sets ``PROMPT_VARIANT`` via ``--export=ALL,PROMPT_VARIANT=...``
so the entire job's process tree picks up the same value. The variant is
also recorded in:

- Each ``keywords.jsonl`` line (``prompt_variant`` field)
- Each row of ``full_experiment_data_{variant}.parquet`` and
  ``dml_results_long_{variant}.parquet`` (``prompt_variant`` /  ``variant`` columns)
- Run dir suffix: ``runs/{engine}_{model}_serp{N}_top{K}_{variant}/``

So you can never accidentally mix biased and neutral coefficients in the
same plot.

---

## 7. Output paths reference

```
geodml_data/
├── data/
│   ├── serp/                                       # cached, do not modify
│   │   └── phase0_top{20,50}_{searxng,ddg}.parquet
│   ├── runs/
│   │   ├── {engine}_{model}_serp{N}_top{K}/        # original cached run dirs (HTML lives here)
│   │   │   └── phase2/html_cache.tar.gz
│   │   └── {engine}_{model}_serp{N}_top{K}_{variant}/   # NEW: port output
│   │       └── phase2/{keywords.jsonl,rankings.csv,.rerank_ckpt.json}
│   ├── features/                                   # NEW: stage B
│   │   └── features_{engine}_top{pool}.parquet
│   ├── main/                                       # NEW: stage C
│   │   └── full_experiment_data_{variant}.parquet
│   └── dml_results/                                # stage D
│       ├── dml_results_long.parquet                # legacy biased reference
│       └── dml_results_long_{variant}.parquet     # NEW
└── ...

interpretability/
└── output/
    ├── ablation_*/              # per (treatment, model) - existing
    ├── saliency_*/              # per (model)            - existing
    ├── probing_*/               # per (model)            - existing
    ├── weights_*/               # per (model)            - existing
    └── plots/
        ├── figure_a_dml_biased.png   # NEW
        ├── figure_a_dml_neutral.png  # NEW
        ├── figure_a_dml_delta.png    # NEW
        ├── figure_a_ablation_*.png   # existing
        ├── figure_b_*.png            # existing
        └── figure_c_probing.png      # existing
```

---

## 8. Troubleshooting

**`Disk quota exceeded` on $HOME**
JUWELS $HOME inode limit (~80k) is easy to blow with a venv. Move the venv
to project space; symlink it back:

```bash
mv ~/.venv ~/.venv.broken
uv venv /p/project1/obdifflearn/vfourel/venvs/geodml --python 3.11
ln -s /p/project1/obdifflearn/vfourel/venvs/geodml ~/.venv
```

Set ``UV_CACHE_DIR``, ``HF_HOME``, ``PIP_CACHE_DIR`` to point at scratch /
project space — see ``important_commands.txt``.

**`HF_HUB_OFFLINE=1` model not found**
Pre-cache on a login node:

```bash
HF_HOME=$SCRATCH/hf_cache python -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.3-70B-Instruct')
snapshot_download('Qwen/Qwen2.5-72B-Instruct')
snapshot_download('all-MiniLM-L6-v2', repo_type='model')
"
```

**`sbatch: error: please specify the job's account`**
JSC's submit filter requires ``--account`` on every sbatch even when ``ALL``
is forwarded. ``_common.sh:chain_resubmit`` reads it from ``JUWELS_ACCOUNT``;
set it in ``.env``.

**Walltime kills mid-keyword**
Every stage uses ``Checkpoint`` + ``--resume``. The chain_resubmit at the
bottom of each sbatch script auto-submits ``afterany:<JID>`` up to
``MAX_ATTEMPTS=6``. Just check ``squeue`` again after the first walltime.

**The biased reproduction gate fails**
Check first: did the cached ``dml_results_long.parquet`` actually use the
``CONFOUNDERS_NEW`` set, or the legacy ``CONFOUNDERS_LEGACY``? See
``interpretability/pipeline/config.py:CONFOUNDERS``. Match the upstream
config or the coefficients will differ by O(0.05+) due to confounder
specification, not a port bug.

**`features.parquet` has tiny T7 row count**
``classify_source_type`` only knows the static ``BRAND_DOMAINS`` /
``EARNED_DOMAINS`` sets. Domains not in either return ``other`` (T7=0).
That's expected; the DML treats ``treat_source_earned == 1`` as the small
treated subgroup. If you need richer T7 labels, extend the static sets in
``interpretability/pipeline/features.py``.

---

## 9. What was NOT ported (and why)

- ``pipeline/gather_data.py`` SERP fetchers (SearXNG, DDG, Google API, etc.) — HTTP, redundant with cached parquets.
- ``pipeline/extract_features.py:fetch_moz_data`` — HTTP + paid API. If you need the Moz columns, pre-cache them on a login node into a parquet and pass via ``--external-features-parquet``.
- ``pipeline/run_phase3.py:llm_extract_treatments`` (LLM-scored T1–T4) — replaced by deterministic parsers. To keep the legacy LLM-scored columns as a robustness check, set ``ENABLE_LLM_TREATMENTS=1`` (port the function then; not done by default).
- ``pipeline/analyze.py:plot_*`` and ``analyze_cross_model.plot_*`` — superseded by ``interpretability.make_figures``.
- ``src/llm_ranker.py`` and ``src/page_features.py`` — legacy duplicates of code that lives in ``pipeline/``.

If you need any of these later, the upstream repo at
``/Users/valerianfourel/Hamburg/GEODML/`` (laptop) or its remote remains the
authoritative source.
