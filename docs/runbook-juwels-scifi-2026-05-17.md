# Runbook — Finish on JUWELS Booster under the `scifi` project (full reconciliation)

End-to-end runbook for the 2026-05-17 work: finish Stage A (rerank) + A'
(order_probe) for all 4 active variants in **full-precision bf16** on the
JUWELS Booster, under the new **scifi** project allocation. Then re-merge
Stage C, refresh Stage D, and push the updated dataset (now with the new
`llm_precision` column) to HuggingFace.

> **TL;DR.** Switch JUWELS account to `scifi`, sync code + data, smoke-test,
> launch `finish_on_gpu.sh`, wait ~2-3 days, run Stages B/C/D locally,
> push consolidated dataset to HF.

---

## 0. What's new since the last runbook

Three changes drive this runbook (commits land alongside it):

1. **`LocalRanker` default flipped to `quantize=False`** (`interpretability/utils.py`).
   The cluster's `--backend local` path now serves bf16 weights across all 4
   GPUs (~140 GB for 70B). Matches the HF Inference endpoint precision; the
   snippet arm becomes scientifically comparable to the RAG arm.
2. **Every JSONL + parquet now carries `llm_backend` + `llm_precision`.**
   Five normalized labels: `bf16-full`, `4bit-nf4`, `api-hf`, `api-openai`,
   `unknown`. Backfilled into all historical data via
   `scripts/backfill_precision.py`. See `precision_label()` at
   `interpretability/pipeline/rerank.py:precision_label`.
3. **New driver `scripts/finish_on_gpu.sh`** mirrors `finish_via_api.sh` but
   submits sbatch jobs through `dispatch_all.sh`. Pre-flight skip-guard in
   `_common.sh:skip_if_at_max` ensures cells already at MAX_KW exit
   instantly without loading the 70B.

---

## 1. Pre-flight on Mac (10 min)

```bash
# 1.1 Commit the new code + backfill artifacts.
cd ~/Hamburg/GEODML_Analysis
git status                                    # 24 modified, 3-5 new
git add interpretability/ scripts/ docs/
git commit -m "feat(precision): bf16 LocalRanker + per-row llm_precision column"
git push origin main

# 1.2 Confirm the local data is backfilled.
.venv/bin/python scripts/backfill_precision.py \
  --root ~/Hamburg/geodml-dataset --include-recent --report-only
# Expect: records_total=41,383  records_patched=0  already_set=41,383

# 1.3 Confirm Stage C parquets carry the new columns.
.venv/bin/python -c "
import pandas as pd
for v in ['biased','neutral','biased_rag','neutral_rag']:
    df = pd.read_parquet(f'$HOME/Hamburg/geodml-dataset/data/main/full_experiment_data_{v}.parquet')
    assert {'llm_backend','llm_precision'} <= set(df.columns), v
    print(f'{v:14s}  precision={df[\"llm_precision\"].value_counts().to_dict()}')
"
```

If you want HuggingFace to have the **current** state first (then re-push
after the cluster work), do an interim push now — see §6.

---

## 2. JUWELS login + scifi project activation (5 min)

```bash
# 2.1 SSH in
ssh juwels   # or:  ssh <username>@juwels-booster.fz-juelich.de

# 2.2 Activate scifi project (sets $PROJECT, $SCRATCH, $USERID, accounting)
jutil env activate -p scifi
# Sanity:
echo "PROJECT=$PROJECT   SCRATCH=$SCRATCH"
sshare -A scifi -u $USER --format=Account,User,RawShares,NormShares,EffectvUsage
# RawShares > 0 means the account is live for you.
```

If `jutil env activate -p scifi` errors with "no such project", your scifi
membership hasn't propagated yet — open a ticket at
https://judoor.fz-juelich.de and don't proceed.

---

## 3. Sync code + venv on JUWELS (15 min — once per machine)

```bash
# 3.1 Land the repo at a known location (use scifi scratch, not $HOME — quota)
cd $SCRATCH
git clone https://github.com/ValerianFourel/GEODML_Analysis.git
cd GEODML_Analysis

# 3.2 Build the venv on the LOGIN node (compute nodes are offline)
module load Stages/2024 GCC Python CUDA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
pip install -r requirements.txt
# Cluster also needs these for GPU + DML:
pip install lightgbm doubleml rank_bm25 sentence-transformers textstat
pip install bitsandbytes accelerate            # for 4-bit fallback if needed
pip install transformers>=4.45 torch           # bf16 70B sharding needs >= 4.45

# 3.3 Quick import sanity
.venv/bin/python -c "
import torch, transformers, accelerate
print(f'torch={torch.__version__}  transformers={transformers.__version__}')
print(f'cuda={torch.cuda.is_available()}  n_gpus={torch.cuda.device_count()}')
"
```

---

## 4. .env on JUWELS — point everything at scifi (5 min)

```bash
cd $SCRATCH/GEODML_Analysis
cp .env.example .env
$EDITOR .env
```

Fill in (delete the example lines you don't use):

```ini
# REQUIRED — scifi project accounting
JUWELS_ACCOUNT=scifi
JUWELS_PROJECT=scifi

# REQUIRED — HF auth for model + dataset download on login node
HF_TOKEN=hf_xxx_read_token

# REQUIRED — where the dataset lives on the cluster
GEODML_DATA_ROOT=/p/scratch/scifi/$USER/geodml_data

# REQUIRED — bf16 by default (matches API). Override at run time if needed.
LOCAL_PRECISION=full

# Optional — what files-per-cell counts as "done" (drives the skip-guard)
MAX_KW=400

# Reference (already in config.py; don't change unless you mean it)
PRIMARY_MODEL=meta-llama/Llama-3.3-70B-Instruct
PROXY_MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_DATASET_REPO=ValerianFourel/geodml-papersize-full
```

Validate:

```bash
set -a; source .env; set +a
env | grep -E "JUWELS_ACCOUNT|JUWELS_PROJECT|GEODML_DATA_ROOT|LOCAL_PRECISION"
# expect: JUWELS_ACCOUNT=scifi, GEODML_DATA_ROOT=/p/scratch/scifi/$USER/geodml_data
```

---

## 5. Pull the dataset snapshot (~30 min, ~7 GB)

```bash
cd $SCRATCH/GEODML_Analysis
mkdir -p "$GEODML_DATA_ROOT"

# 5.1 Download the latest HF snapshot (includes the precision-backfilled
#     JSONLs + the 4 merged Stage C parquets with llm_precision column).
.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    'ValerianFourel/geodml-papersize-full',
    repo_type='dataset',
    local_dir=os.path.expandvars('$GEODML_DATA_ROOT/..'),
    local_dir_use_symlinks=False,
    allow_patterns=['data/**', 'PROVENANCE.md', 'README.md'],
)
print('snapshot OK')
"

# 5.2 Sanity-check the precision columns are in the dataset
.venv/bin/python -c "
import pandas as pd, os
ROOT = os.path.expandvars('$GEODML_DATA_ROOT/data/main')
for v in ['biased','neutral','biased_rag','neutral_rag']:
    df = pd.read_parquet(f'{ROOT}/full_experiment_data_{v}.parquet')
    print(f'{v:14s}  precision={df[\"llm_precision\"].value_counts().to_dict()}')
"

# 5.3 Pre-populate the HF model cache so HF_HUB_OFFLINE=1 on compute nodes works.
#     ~140 GB per 70B model — make sure $HF_HOME has room ($SCRATCH/hf_cache is the
#     _common.sh default).
export HF_HOME=$SCRATCH/hf_cache
mkdir -p "$HF_HOME"
.venv/bin/python -c "
from huggingface_hub import snapshot_download
for m in ['meta-llama/Llama-3.3-70B-Instruct', 'Qwen/Qwen2.5-72B-Instruct',
          'meta-llama/Llama-3.1-8B-Instruct']:
    print('downloading', m); snapshot_download(m, cache_dir='$HF_HOME')
"
```

If 5.3 takes too long (>2 hr), let it run via tmux / nohup. Compute nodes
need these files local to `$HF_HOME` to load.

---

## 6. (Optional) Push the current Mac-side state to HF before the cluster run

You wanted the existing data on HF with the new precision column. This is
the right time to do that — before the cluster overwrites it.

On Mac:

```bash
cd ~/Hamburg/GEODML_Analysis

# 6.1 Rebuild the consolidated tree with real files (HF won't follow symlinks)
COPY_HTML=1 COPY_DATA=1 FORCE=1 bash scripts/build_dataset_mirror.sh

# 6.2 Push. The build script wrote a push_to_hf.sh into the dataset root.
hf auth login                                  # paste a WRITE-scoped token
REPO=ValerianFourel/geodml-papersize-full \
  bash ~/Hamburg/geodml-dataset/push_to_hf.sh
```

After this, ~/Hamburg/geodml-dataset/README.md + PROVENANCE.md on HF
document the `llm_precision` column with the current breakdown (snippet =
4bit-nf4, RAG = api-hf).

The cluster run in §7-§9 will produce a NEW snippet arm in bf16-full;
§10 re-pushes so the dataset shows the final state.

---

## 7. Smoke test on develbooster (5 keywords, ~15 min)

```bash
cd $SCRATCH/GEODML_Analysis
set -a; source .env; set +a

# 7.1 Submit a smoke rerank: 1 cell, 5 keywords, bf16, develbooster partition
MAX_KW=5 LOCAL_PRECISION=full \
  ./scripts/slurm/dispatch_all.sh --smoke \
    --only rerank --variant biased \
    --models meta-llama/Llama-3.3-70B-Instruct \
    --engines searxng --pools 20

# 7.2 Watch the job
squeue -u $USER --format='%.10i %.32j %.2t %.10M' | head
JID=$(squeue -u $USER -h -o '%i' | head -1)
tail -f logs/rerank-*-$JID*.out
```

**Pass criteria** (in the log):

- `[load] CUDA_VISIBLE_DEVICES gives 4 GPUs; max_memory={0: 75GiB, ...}`
- `[LocalRanker] model=meta-llama/Llama-3.3-70B-Instruct precision=bf16-full`
- `[LocalRanker] hf_device_map: ... 4 devices` (model sharded across all 4)
- `[LocalRanker]   cuda:N allocated=~35 GiB` per GPU
- `[rerank] backend=local model=... variant=biased precision=full`
- JSONL record on disk contains `"llm_parameters": {... "precision": "bf16-full"}`

Verify the JSONL:

```bash
.venv/bin/python -c "
import json
p='$GEODML_DATA_ROOT/data/runs/searxng_Llama-3.3-70B-Instruct_serp20_top10_biased/phase2/keywords.jsonl'
with open(p) as f:
    rec = json.loads(f.readline())
assert rec['llm_parameters']['precision'] == 'bf16-full', rec['llm_parameters']
print('smoke OK — record carries precision=bf16-full')
"
```

If the smoke run uses 4-bit by mistake, the log will show
`precision=4bit-nf4` and the JSONL will carry it — *do not proceed* until
you see `bf16-full`. The most common cause is `LOCAL_PRECISION` not being
exported into the sbatch (the `dispatch_all.sh` emit helper injects it; if
you submitted by hand, add `LOCAL_PRECISION=full` to your `--export`).

---

## 8. Full launch (~2-3 days)

```bash
cd $SCRATCH/GEODML_Analysis
set -a; source .env; set +a

# 8.1 The big one: snippet rerank redo + RAG order_probe finish.
MAX_KW=400 LOCAL_PRECISION=full \
  ./scripts/finish_on_gpu.sh \
    2>&1 | tee logs/finish_on_gpu_$(date +%Y%m%d_%H%M%S).log

# Default scope:
#   * Phase 2: rerank biased + neutral × 2 models × 2 engines × 2 pools = 16 jobs
#   * Phase 3: order_probe (biased, neutral, biased_rag, neutral_rag) × 64 jobs
#     The skip-guard auto-skips cells already at MAX_KW=400 — so the 6 API'd
#     RAG cells exit instantly and only the 26 pending ones actually run.

# 8.2 If you ALSO want to redo RAG rerank in bf16 (strictest reconciliation):
INCLUDE_RAG_REDO=1 MAX_KW=400 LOCAL_PRECISION=full \
  ./scripts/finish_on_gpu.sh

# 8.3 If you ALSO want to close the Stage F gaps (probing for neutral; all four
#     F methods for _rag — see docs/long-term-project-arc.md §7.1):
INCLUDE_F_GAPS=1 INCLUDE_RAG_REDO=1 MAX_KW=400 LOCAL_PRECISION=full \
  ./scripts/finish_on_gpu.sh
```

---

## 9. Monitor (intermittent over the next 2-3 days)

```bash
# 9.1 Queue state
watch -n 30 'squeue -u $USER --format="%.10i %.32j %.2t %.10M %.10l" | head -40'

# 9.2 Live tail of the most-recently-modified log
tail -f $(ls -t logs/*.out | head -1)

# 9.3 Cell-by-cell progress
.venv/bin/python scripts/audit_status.py | tail -50

# 9.4 Quick precision audit — every newly-written JSONL should carry bf16-full
.venv/bin/python -c "
import json, glob, os
root = os.environ['GEODML_DATA_ROOT']
for p in sorted(glob.glob(f'{root}/data/runs/*_biased/phase2/keywords.jsonl') +
                glob.glob(f'{root}/data/runs/*_neutral/phase2/keywords.jsonl')):
    with open(p) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    prec = {r['llm_parameters'].get('precision') for r in recs}
    print(f'{len(recs):>5}  {prec}  {p.split(chr(47))[-3]}')
" | head -20
```

Expected: after the snippet rerank cells finish (~6h each on bf16), the
precision set should be `{'bf16-full'}`. Any cell showing both `4bit-nf4`
and `bf16-full` means the cell didn't get fully truncated before the redo
— see §11 for how to recover.

---

## 10. After all cluster jobs finish

### 10.1 Run Stage B/C/D on the login node (~30 min, no GPU needed)

```bash
cd $SCRATCH/GEODML_Analysis
set -a; source .env; set +a
bash scripts/continue_pipeline.sh
# Runs: Stage B (cached), Stage C (merge × 4 variants), Stage D (DML × 4),
# make_figures.py, order_probe_analyze.py.
```

### 10.2 Verify the precision crossover

```bash
.venv/bin/python -c "
import pandas as pd, os
ROOT = os.environ['GEODML_DATA_ROOT']
for v in ['biased','neutral','biased_rag','neutral_rag']:
    df = pd.read_parquet(f'{ROOT}/data/main/full_experiment_data_{v}.parquet')
    print(f'{v:14s} {df[\"llm_precision\"].value_counts().to_dict()}')
"
# Expected after the full bf16 redo:
#   biased         {'bf16-full': N1}
#   neutral        {'bf16-full': N2}
#   biased_rag     {'api-hf': N3}                  ← if INCLUDE_RAG_REDO unset
#                  {'bf16-full': N3'} or mix       ← if INCLUDE_RAG_REDO=1
#   neutral_rag    same as biased_rag
```

### 10.3 Run audit

```bash
.venv/bin/python scripts/audit_pipeline.py 2>&1 | tail -40
.venv/bin/python scripts/audit_status.py | tail -20
# Expect:
#   Stage A  32/32 (or 16/32 if RAG kept)
#   Stage A' 96/96 (or 64/96 if RAG kept)
#   Stage B  4/4
#   Stage C  4/4 (or 6/6 if you re-merged passage too)
#   Stage D  4/4
#   Stage F  80/120 if no F gaps; 120/120 if INCLUDE_F_GAPS=1 succeeded
```

### 10.4 Headline coefficient sanity (catches silent regressions)

```bash
.venv/bin/python -c "
import pandas as pd, os
ROOT = os.environ['GEODML_DATA_ROOT']
df = pd.read_parquet(f'{ROOT}/data/dml_results/dml_results_long_biased.parquet')
row = df.query('subset==\"POOLED\" and method==\"plr\" and learner==\"lgbm\" '
               'and outcome==\"rank_delta\" and treatment==\"T7_source_earned\"').iloc[0]
print(f'T7 biased POOLED.plr.lgbm.rank_delta: coef={row[\"coef\"]:+.3f}  '
      f'p={row[\"p_val\"]:.4f}')
# Pre-bf16 (4-bit): coef ≈ -1.607***
# Post-bf16:        expect coef within ±15% (~-1.4 to -1.85), still p<0.001
"
```

If T7 is dramatically different (wrong sign, p>0.05, magnitude <0.5), the
bf16 redo is producing materially different LLM behavior. Investigate
before pushing to HF — this is a paper-worthy result either way but
should be documented.

---

## 11. Sync back to Mac + push to HuggingFace (final state)

### 11.1 Pull cluster data back to Mac

```bash
# On Mac
cd ~/Hamburg/GEODML_Analysis
rsync -avz --partial --info=progress2 \
  juwels:/p/scratch/scifi/$USER/geodml_data/data/ \
  ~/Hamburg/geodml-dataset/data/
```

### 11.2 Rebuild + push the consolidated dataset

```bash
# Rebuild with real files (HF needs them — symlinks don't transfer)
cd ~/Hamburg/GEODML_Analysis
COPY_HTML=1 COPY_DATA=1 FORCE=1 bash scripts/build_dataset_mirror.sh

# Push
hf auth login   # write-scoped token if not already
REPO=ValerianFourel/geodml-papersize-full \
  bash ~/Hamburg/geodml-dataset/push_to_hf.sh
```

The pushed dataset README/PROVENANCE now document the new
`llm_precision` column with the post-bf16 breakdown.

---

## 12. Failure modes + recovery

### 12.1 Smoke test shows `precision=4bit-nf4`
- Cause: `LOCAL_PRECISION` not exported. Check
  `cat $GEODML_DATA_ROOT/data/runs/*/phase2/keywords.jsonl | head -1 | jq .llm_parameters`.
- Fix: `LOCAL_PRECISION=full sbatch --export=ALL,...,LOCAL_PRECISION=full ...`.
  Re-submit. The next record will be `bf16-full`. Discard the contaminated cell
  (rename `phase2/` and re-run).

### 12.2 OOM on Qwen-72B bf16
- Symptom: `torch.cuda.OutOfMemoryError` after device_map allocation.
  Qwen-72B bf16 is right at the edge of 4×80 GB with the 2400-char RAG prompt.
- Fix: drop `--top-k-rag` to 2 (chunks/page), reducing prompt length:
  ```bash
  # Patch the rerank.sbatch invocation:
  sed -i 's/--top-k-rag 3/--top-k-rag 2/' scripts/slurm/run_rerank.sbatch
  ```
  Or escalate to `--gres=gpu:8` on a single node if scifi allocation has it.

### 12.3 `jutil env activate -p scifi` fails
- Cause: scifi project not yet propagated to your account.
- Fix: ticket at https://judoor.fz-juelich.de.
- Workaround: run on the OLD account by overriding for one launch:
  `JUWELS_ACCOUNT=old_project ./scripts/finish_on_gpu.sh` — but ALL accounting
  goes to old_project, which is what you wanted to avoid.

### 12.4 Job killed by walltime
- The `chain_resubmit` helper in `_common.sh` auto-submits a continuation
  with `--dependency=afterany:` up to `MAX_ATTEMPTS=6`. Just wait.
- If it gives up: `tail logs/<jobid>.out` for the real failure, fix, then
  `MAX_ATTEMPTS=12 INCLUDE_RAG_REDO=... ./scripts/finish_on_gpu.sh` for one
  more pass — skip-guard will pick up exactly where the chain left off.

### 12.5 Mixed precision contamination in a cell
- Symptom: `audit precision` check shows a cell with records under both
  `4bit-nf4` and `bf16-full`.
- Cause: bf16 redo resumed without truncating the old 4-bit JSONL first
  (the rerank.py `--resume` flag preserves existing records).
- Fix: archive + truncate + re-run:
  ```bash
  V=biased
  for D in $GEODML_DATA_ROOT/data/runs/*_${V}/phase2 ; do
    mv "$D/keywords.jsonl" "$D/keywords.jsonl.bak_$(date +%s)"
    rm -f "$D/.rerank_ckpt.json"
  done
  # then resubmit just that variant
  LOCAL_PRECISION=full ./scripts/slurm/dispatch_all.sh \
    --only rerank --variant $V
  ```

---

## 13. Total budget

| Phase | Wall time | GPU-hours (scifi accounting) |
|---|---|---|
| Smoke test | 15 min | 0.25 |
| §8.1 snippet redo (16 cells) | ~2 days | ~100 |
| §8.2 RAG redo (16 cells, optional) | ~2 days | ~100 |
| §8.3 Stage F gaps (~10 jobs, optional) | ~1 day | ~30 |
| Stage B/C/D + figures (login node, no GPU) | 30 min | 0 |
| HF push | 1 hr | 0 |
| **Total (default)** | **~3 days** | **~100 GPU-hr** |
| **Total (with RAG redo + F gaps)** | **~5 days** | **~230 GPU-hr** |

scifi should comfortably absorb 100-230 GPU-hours; check with
`sshare -A scifi -u $USER` before launching the strictest variant.

---

## 14. Reference — what changed in code (pointers for future readers)

| File | Change |
|---|---|
| `interpretability/utils.py:412` | `LocalRanker.__init__(model, quantize=False, ...)` — bf16 by default. |
| `interpretability/utils.py:285` | `multi_gpu_load_kwargs(quantize=False)` now sets `torch_dtype=bfloat16` on CUDA. |
| `interpretability/utils.py:528` | `make_ranker(backend, model, *, precision)` — honors `LOCAL_PRECISION` env. |
| `interpretability/pipeline/rerank.py:precision_label` | Backend×precision → canonical string. |
| `interpretability/pipeline/rerank.py:rank_one_keyword` | Accepts + records `backend, precision`. |
| `interpretability/pipeline/order_probe.py:228` | Same pass-through. |
| `interpretability/pipeline/merge.py:130` | Adds `llm_backend, llm_precision` to Stage C parquet. |
| `interpretability/weight_analysis.py:204` | Reads `--variant` (env `PROMPT_VARIANT`); fixes the silent-biased bug. |
| `scripts/slurm/_common.sh:67` | `skip_if_at_max` pre-flight short-circuit. |
| `scripts/slurm/dispatch_all.sh:54` | `ORDER_PROBE_VARIANTS=(biased neutral biased_rag neutral_rag)`. |
| `scripts/slurm/dispatch_all.sh:107` | `LOCAL_PRECISION` injected into every child sbatch. |
| `scripts/finish_on_gpu.sh` | **NEW** — GPU equivalent of `finish_via_api.sh`. |
| `scripts/backfill_precision.py` | **NEW** — patches historical JSONLs with inferred backend/precision. |
| `scripts/build_dataset_mirror.sh:240` | README + PROVENANCE document `llm_precision` column. |

End.
