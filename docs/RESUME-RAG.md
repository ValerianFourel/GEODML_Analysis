# Resume the RAG run (one-page cheat sheet)

For full context: `docs/work-log-2026-05-08.md`.

## Where we stopped (2026-05-08)

- **Phase 2 rerank: ✅ all 16 cells done** (6,919 records, 0 fallbacks).
- **Phase 3 order_probe: ⏸️ stopped at cell 7/32** (HF Inference 402:
  monthly credits depleted).
  - 6 cells fully done (400 records each)
  - cell 7 partial (23 real records preserved; 113 fallbacks dropped, backup at `*.pre402_bak`)
  - 25 cells pending

## Step 1 — top up HF Inference credits

Browser → https://huggingface.co/settings/billing → either add a payment
method or buy a prepaid pack.

Sanity-check it landed:

```bash
cd ~/Hamburg/GEODML_Analysis
.venv/bin/python -c "
import os; from dotenv import load_dotenv; load_dotenv('.env')
from huggingface_hub import InferenceClient
c = InferenceClient(token=os.environ['HF_TOKEN'], timeout=30)
r = c.chat_completion(model='meta-llama/Llama-3.3-70B-Instruct',
    messages=[{'role':'user','content':'OK'}], max_tokens=5)
print('Llama:', r.choices[0].message.content)
"
```

If you see `Llama: OK` (or similar), credits are live.
If you see 402, wait and retry.

## Step 2 — make sure no zombie process

```bash
ps aux | grep -E "interpretability\.pipeline\.(rerank|order_probe)" | grep -v grep
```

Should be empty. If not: `pkill -f interpretability.pipeline`.

## Step 3 — launch resume

```bash
cd ~/Hamburg/GEODML_Analysis
source .venv/bin/activate
export PATH=$PWD/.venv/bin:$PATH
LOG=logs/rag_resume_$(date +%Y%m%d_%H%M%S).log
echo "Log: $LOG"
nohup bash -c '
  set -a; source .env; set +a
  export GEODML_DATA_ROOT=$HOME/Hamburg/geodml-dataset
  export PATH=$PWD/.venv/bin:$PATH
  MAX_KW=400 VARIANTS_AUG="biased_rag neutral_rag" \
    SKIP_BCD=1 \
    bash scripts/finish_via_api.sh
' > "$LOG" 2>&1 &
echo "PID=$!"
```

What it does:

- **Phase 1, 1.5, 2 — all skipped** (everything's already cached / built).
- **Phase 3 — runs cells 7-32** (first 6 already done, skipped via guard).
- **Stage B/C/D — skipped** via `SKIP_BCD=1` (run those after Phase 3 finishes).

ETA: ~9 hr (~1 hr Llama cells, ~8 hr Qwen cells). Cost: ~$15.

## Step 4 — watch progress

```bash
# tail the live log
tail -f logs/rag_resume_*.log

# count finished order_probe cells
ls -la ~/Hamburg/geodml-dataset/data/order_probe/*_rag_seed*.jsonl | wc -l
# should grow from 7 → 32

# spot-check no fallbacks
.venv/bin/python -c "
import json, glob
for f in sorted(glob.glob('$HOME/Hamburg/geodml-dataset/data/order_probe/*_rag_seed*.jsonl')):
    recs = [json.loads(l) for l in open(f) if l.strip()]
    fb = sum(1 for r in recs if r.get('used_fallback'))
    print(f'{len(recs):>4} {fb} fallbacks  {f.split(chr(47))[-1]}')
"
```

## Step 5 — after Phase 3 finishes, run B/C/D + figures

```bash
unset SKIP_BCD
bash scripts/continue_pipeline.sh
```

This runs:
- Stage B (variant-agnostic features, fast)
- Stage C (`merge.py --variant biased_rag` + `--variant neutral_rag`)
- Stage D (`dml.py --variant biased_rag` + `--variant neutral_rag`)
- `make_figures.py` (will pick up the new variants because
  `KNOWN_VARIANTS` includes them now)
- `order_probe_analyze.py` (re-aggregates `order_probe_summary.parquet`)

Then `python scripts/audit_pipeline.py` should report 6 variants (was 4)
with `Stage A 24/48` for the snippet+RAG variants (Qwen passage cells were
archived) and `Stage A' 96/96` (the 32 new RAG seeds joining the 64 snippet
seeds).

## If something goes wrong

- **402 again mid-run**: same fix — top up, kill the run, relaunch. Skip-guard
  will pick up where it left off. `.pre402_bak` backups protect against
  fallback contamination.
- **NaN errors come back**: shouldn't — the guard at `rerank.py:_serp_to_results`
  is in place.
- **Network wedge / hang**: the 90s timeout on `InferenceRanker` should
  trigger; if not, kill the python process and relaunch.
- **Want a smaller smoke first**: prepend `MAX_KW=20` and let it run. Skip-
  guard interprets MAX_KW as the threshold, so existing 400-records cells
  stay alone.

## Cost-saver (optional)

If Qwen cells stay slow, set `MAX_KW=200` for Phase 3 — halves wall time
and cost while keeping enough samples for the order_probe Jaccard / OAK
metrics. Cells already at 400 stay; new cells run to 200. Analysis pipeline
handles mixed counts via per-keyword aggregation.

```bash
MAX_KW=200 VARIANTS_AUG="biased_rag neutral_rag" SKIP_BCD=1 \
  bash scripts/finish_via_api.sh
```

## Before you publish, rotate credentials

Both keys appear in this conversation transcript and in `.env` (which is
gitignored, but still). Rotate after the run:

- HF: https://huggingface.co/settings/tokens
- OpenAI: https://platform.openai.com/api-keys
