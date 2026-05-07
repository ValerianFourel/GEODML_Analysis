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
