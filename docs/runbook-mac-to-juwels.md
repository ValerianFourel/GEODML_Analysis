# Runbook — finish on macOS, hand off to JUWELS GPU

End-to-end checklist for the remaining work. Phases 1–4 run on your Mac
and produce Stages B/C/D + figures. Phase 5 finishes Stage F (6 probing
cells) on JUWELS.

## Phase 1 — Mac setup (one-time, ~10 min)

```bash
# 1.1 brew prereq for LightGBM
brew install libomp

# 1.2 clone + venv (Python 3.12 recommended; avoid 3.13)
git clone https://github.com/ValerianFourel/GEODML_Analysis.git
cd GEODML_Analysis
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel

# 1.3 install Python deps (requirements.txt + the four it forgets)
pip install -r requirements.txt
pip install lightgbm doubleml rank_bm25 sentence-transformers textstat

# 1.4 sanity check
python -c "
import pandas, sklearn, lightgbm, doubleml, sentence_transformers
import bs4, lxml, rank_bm25, textstat, huggingface_hub
print('deps OK')
"

# 1.5 HF token — READ-only is enough for downloading
cp .env.example .env
$EDITOR .env   # paste HF_TOKEN=hf_xxx_read_token
```

## Phase 2 — Pull the snapshot (~15 min, ~3 GB)

```bash
# 2.1 grab the two zips you uploaded from the cluster
mkdir -p archives
hf download ValerianFourel/geodml-papersize \
  archives/geodml_data_.zip archives/interpretability_.zip \
  --repo-type dataset --local-dir .

# 2.2 unzip in place
unzip -q archives/geodml_data_.zip
unzip -q archives/interpretability_.zip

# 2.3 sync the missing html_cache.tar.gz files from HF (only fetches what
#     isn't already on disk — ~800 MB of searxng HTML)
python scripts/sync_data.py

# 2.4 export the data root so the pipeline scripts find it
export GEODML_DATA_ROOT="$PWD/geodml_data"

# 2.5 verify the audit is happy with the local state
python scripts/audit_pipeline.py | tail -3
# Expect: Stage A 24/32  Stage B 0/4  Stage C 0/4  Stage D 0/4  Stage F 74/80  Order probe 48/64
```

## Phase 3 — Run Stage B/C/D + figures (~30–60 min, CPU)

```bash
# 3.1 (optional) smoke-test with a tiny keyword cap first
FEATURES_MAX_KW=50 bash scripts/continue_pipeline.sh
#   - confirms imports, html_cache, embedder loading
#   - if it finishes cleanly, run the real thing:

# 3.2 the real run
bash scripts/continue_pipeline.sh
```

What you'll see at the end:
- A new audit showing `Stage B 2/4 · C 4/4 · D 4/4` (B is 2/4 because ddg
  has no HTML — that's by design).
- A zip at `archives/local_results_<YYYYMMDD-HHMM>.zip` (typically <100 MB).
- The exact `hf upload` command printed for Phase 4.

## Phase 4 — Push results back to HF (~2 min)

```bash
# 4.1 get a WRITE-scoped HF token at https://huggingface.co/settings/tokens
#     (don't reuse the read one). DO NOT paste it on the command line.
huggingface-cli login   # paste the write token at the prompt

# 4.2 paste the upload command the script printed at end of Phase 3,
#     e.g.:
RESULT_ZIP=archives/local_results_20260507-1530.zip
hf upload ValerianFourel/geodml-papersize \
  "$RESULT_ZIP" \
  "$RESULT_ZIP" \
  --repo-type dataset \
  --commit-message "local Stage B/C/D"

# 4.3 verify it landed
hf api datasets/ValerianFourel/geodml-papersize/tree/main/archives \
  | python -m json.tool | head -30
```

Single-file commit → won't trip the 500 you saw earlier.

## Phase 4.5 (recommended) — Finish Stage A 32/32 via API

Use this when you've lost cluster GPU access and want to fill **all 8
missing rerank cells + 16 order-probe cells**. With `LOCAL_HTML_SOURCE`
auto-detected, no scraping is needed — the legacy upstream experiment dirs
on your Mac have ddg HTML for both pool=20 and pool=50.

End state: Stage A **24/32 → 32/32**, Stage A' **48/64 → 64/64**,
Stage B **2/4 → 4/4** (ddg now joins via the same local caches), C 4/4,
D 4/4. Only Stage F probing (2/8 → needs GPU) stays unfinished.

```bash
# 4.5.1 install the OpenAI client (used by OpenAIRanker)
pip install openai

# 4.5.2 sign up for an inference provider that hosts both Llama-3.3-70B and
#       Qwen2.5-72B. Recommended:
#         DeepInfra      — cheapest (~$10 total)
#         Together AI    — has both as -Turbo variants
#         Fireworks AI   — fast, similar price
#       Add ~$15 in credit.

# 4.5.3 set provider env (DeepInfra shown — see the case block in the
#       script for the other presets)
export PROVIDER=deepinfra
export OPENAI_API_KEY=<your_deepinfra_key>

# 4.5.4 confirm LOCAL_HTML_SOURCE will be found (auto-detect handles
#       ~/Hamburg/GEODML/paperSizeExperiment/output by default)
ls /Users/valerianfourel/Hamburg/GEODML/paperSizeExperiment/output/duckduckgo_Qwen2.5-72B-Instruct_serp20_top10/html_cache | head
# expect: 0009ef7ee16573b5.html  ...  (~6,000 files per ddg cache, ~8 GB total
#                                       across all 4 ddg cells)

# 4.5.5 run. Idempotent — re-runs skip already-done cells.
bash scripts/finish_via_api.sh
```

What the script does:

1. **Phase 1 — Symlink local html_caches.** Walks
   `$LOCAL_HTML_SOURCE/<engine>_<Model>_serp<N>_top10/html_cache/` for
   {searxng, duckduckgo} × {Llama, Qwen} × {20, 50} = 8 caches. Renames
   `duckduckgo` → `ddg` and links each one under
   `data/runs/<engine>_<Model>_serp<N>_top10/phase2/html_cache`. Both
   Stage B and rerank's `_build_passage_map` then find them transparently.

2. **Phase 2 — Rerank.** All 8 missing cells:
   `ddg × {pool=20, pool=50} × {biased_passage, neutral_passage} ×
   {Llama, Qwen}`. Each call hits the OpenAI-compatible API with
   `temperature=0.1, max_tokens=500` (read from `config.py` — matches
   cluster). Resumable.

3. **Phase 3 — Order probe.** Same 8 cells × 2 seeds (42, 123) = 16 cells.
   Re-aggregates `order_probe_summary.parquet` at the end.

4. **Phase 4 — Stages B/C/D + figures + package.** Delegates to
   `continue_pipeline.sh`. Stage B now includes ddg cells (since their
   html_caches are linked). C and D pick up the new rerank rows
   automatically.

### Cost & wall time

Per pool (covers 4 cells = 2 variants × 2 models, ×2 for order-probe seeds):

| Pool | Keywords | In tokens / call | Total in (rerank + probe) | DeepInfra cost |
|---|---|---|---|---|
| 20 | 79 | ~5k | ~5M | ~$2 |
| 50 | 600 | ~13k | ~94M | ~$25 |

Roughly **$15-30 total on DeepInfra**, ~1-2 hours wall (rate-limit bound).

### Smoke test before paying

```bash
MAX_KW=20 bash scripts/finish_via_api.sh
# Verify a few jsonl rows landed under
#   data/runs/ddg_*_serp{20,50}_top10_*_passage/phase2/keywords.jsonl
# Spot-check the rerank output looks reasonable, then re-run without MAX_KW:
bash scripts/finish_via_api.sh
```

### Temperature contract

Both `OpenAIRanker.rank` and `LocalRanker.rank` read `C.LLM_TEMPERATURE` and
`C.LLM_MAX_TOKENS` from `interpretability/pipeline/config.py`. The script
asserts those values are still `0.1` and `500` respectively before doing
any API calls. If `config.py` has drifted, the script aborts.

After 4.5 finishes, jump to Phase 4 (zip + upload). The package now
contains both the new rerank/order-probe cells and Stage B/C/D outputs.
Phase 5 (probing) waits until you have GPU access back.

## Phase 5 — Finish Stage F on JUWELS (probing, GPU)

The 6 missing probing cells (`neutral`, `biased_passage`, `neutral_passage`
× {Llama-3.3-70B, Qwen2.5-72B}) need the model loaded with
`output_hidden_states=True`. No API exposes that → JUWELS only.

```bash
# 5.1 ssh in, pull the new commits (continue_pipeline.sh + this runbook)
ssh fourel1@jwlogin08.fz-juelich.de
cd /p/project1/obdifflearn/vfourel/GEODML_Analysis
git pull origin main

# 5.2 (optional) pull your local results zip back onto the cluster, so
#     the cluster snapshot is consistent with what's on HF
python scripts/sync_data.py
#     The new archives/local_results_*.zip will be downloaded; if you want
#     to merge it into geodml_data/, do:
unzip -o archives/local_results_*.zip -d /tmp/merge && \
  rsync -a /tmp/merge/geodml_data/ "$GEODML_DATA_ROOT/"

# 5.3 submit the 6 probing jobs (4×A100-40G each, ~4–10 h wall)
for MODEL in meta-llama/Llama-3.3-70B-Instruct Qwen/Qwen2.5-72B-Instruct; do
  for V in neutral biased_passage neutral_passage; do
    sbatch --account="$JUWELS_ACCOUNT" \
      --export=ALL,MODEL="$MODEL",PROMPT_VARIANT="$V" \
      scripts/slurm/run_probing.sbatch
  done
done
squeue -u $USER

# 5.4 once they all complete, audit + re-package + upload from JUWELS
python scripts/audit_pipeline.py    # expect Stage F 80/80 now
zip -r interpretability_$(date +%Y%m%d).zip interpretability/output \
  -x '*/t7_chunks*' '*/__pycache__/*' '*/checkpoint_*.json' '*.tmp.npz'
hf upload ValerianFourel/geodml-papersize \
  interpretability_$(date +%Y%m%d).zip \
  archives/interpretability_$(date +%Y%m%d).zip \
  --repo-type dataset --commit-message "Stage F probing complete"
```

## Done state

- HF dataset `ValerianFourel/geodml-papersize` has, under `archives/`:
  - `geodml_data_.zip` — original cluster snapshot (Phase 0, already there)
  - `interpretability_.zip` — original Stage F snapshot (already there)
  - `local_results_<date>.zip` — Stages B/C/D from your Mac (Phase 4)
  - `interpretability_<date>.zip` — final Stage F with all probing cells (Phase 5)
- Audit on either box shows `Stage F 80/80`. Stage A and Order probe
  remain at 24/32 and 48/64 respectively — those eight ddg-passage cells
  are permanently unfilled (no ddg HTML exists anywhere; not API-runnable).
- Paper figures are at `interpretability/output/plots/figure_{a,b,c}_*.png`
  on whichever box you ran `make_figures` last.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `lightgbm` import error: libomp not found | `brew install libomp`; if still failing, add `$(brew --prefix libomp)/lib` to `DYLD_LIBRARY_PATH` |
| `pip install lxml` fails | `pip install -U pip` first; then retry |
| `sync_data.py` says HF_TOKEN not set | `cat .env` — confirm `HF_TOKEN=hf_...` line is there, no quotes, no spaces |
| `hf upload` 500 error | The script's single-file commit shouldn't 500. If it does, retry — HF transient. |
| `continue_pipeline.sh` aborts at "config.py drifted" | Check git status — you have local edits in `interpretability/pipeline/config.py`. Reset them or fix the values back to `0.1` / `500` before running. |
| Probing job stops after 24 h | The `chain_resubmit` in `run_probing.sbatch` will auto-resubmit on time-out; verify with `squeue` |
