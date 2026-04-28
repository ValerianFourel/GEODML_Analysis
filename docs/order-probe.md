# Order-sensitivity probe — operator runbook

## The question

Is the LLM rerank doing real ranking, or is it anchoring on the order in which
SERP candidates are fed to it? If the top-K output is invariant to a random
permutation of the input list, the rerank reflects substantive judgments. If
overlap collapses under reordering, part of the rerank signal — and therefore
part of every DML coefficient computed downstream — is an order-anchoring
artefact.

## Design

For each `(model, engine, pool, variant)` cell of the main grid, we run
`rank_one_keyword()` two extra times with the candidate list shuffled by a
per-keyword RNG (`random.Random("{seed}::{keyword}")`). Compared to the
canonical "original" rerank already produced by Stage A, this gives three
orderings per cell: `orig`, `seed42`, `seed123`.

The analyzer computes pairwise top-K overlap (Jaccard and overlap@K at
K∈{3,5,10}) for every keyword present in all three files.

| Setting     | Value                                         |
|-------------|-----------------------------------------------|
| Variants    | biased + neutral                              |
| Models      | Llama-3.3-70B-Instruct, Qwen2.5-72B-Instruct  |
| Engines     | searxng, ddg                                  |
| Pools       | 20, 50                                        |
| Seeds       | 42, 123                                       |
| Temperature | 0.1 (matches main rerank)                     |
| Keywords    | full set                                      |
| Total       | 32 jobs ≈ 32 000 LLM calls ≈ 64 GPU-h         |

The shuffle-vs-shuffle pair (`seed42_vs_seed123`) is a noise-floor: it tells
us how much overlap two arbitrary orderings *can* produce given fixed inputs.
Subtracting it from `orig_vs_seedX` isolates the order-anchoring component.

## Files

| Path                                                  | Purpose                                 |
|-------------------------------------------------------|-----------------------------------------|
| `interpretability/pipeline/order_probe.py`            | Stage A' — shuffles + reruns rerank     |
| `interpretability/pipeline/order_probe_analyze.py`    | Pairwise overlap → summary parquet      |
| `scripts/slurm/run_order_probe.sbatch`                | One job per cell × seed (chain-resubmit)|
| `scripts/slurm/dispatch_all.sh --only order_probe`    | Fan out 32 jobs                         |
| `scripts/audit_pipeline.py`                           | Includes Stage A' section + headline    |
| `interpretability/make_figures.py: figure_e_*`        | Boxplots + biased-vs-neutral Δ          |

Outputs (under `$GEODML_DATA_ROOT/data/order_probe/`):

```
{run_id}_seed{S}.jsonl        # one row per keyword, same envelope as
                              # keywords.jsonl + seed/input_order_perm/original_order
order_probe_summary.parquet   # variant, model, engine, pool, keyword, K,
                              # ordering_pair, jaccard, overlap_at_k, n_a, n_b
```

`run_id` follows the same convention as the main pipeline:
`{engine}_{ModelTag}_serp{N}_top{K}_{variant}`.

## Run on JUWELS

```bash
# 0. Pull, activate project, set scratch.
git pull origin main
jutil env activate -p obdifflearn
export GEODML_DATA_ROOT=$SCRATCH/geodml_data

# 1. (Optional) snapshot before
python scripts/audit_pipeline.py --save audits/before_order_probe.json

# 2. Smoke first — one cell, 20 keywords, mock ranker (no GPU needed).
python -m interpretability.pipeline.order_probe \
    --variant neutral --engine searxng --pool 20 --seed 42 \
    --max-keywords 20 --smoke --resume \
    --out-run-id smoke_searxng_serp20_top10_neutral

# Inspect:
ls -lh $GEODML_DATA_ROOT/data/order_probe/smoke_*_seed42.jsonl
head -1 $GEODML_DATA_ROOT/data/order_probe/smoke_*_seed42.jsonl | python -m json.tool

# 3. Real cluster smoke — one real cell, capped at 50 keywords.
sbatch --account=$JUWELS_ACCOUNT \
  --export=ALL,MODEL=Qwen/Qwen2.5-72B-Instruct,ENGINE=searxng,POOL=20,PROMPT_VARIANT=neutral,SEED=42,MAX_KEYWORDS=50 \
  scripts/slurm/run_order_probe.sbatch

# 4. Full grid (32 jobs, ~2h GPU each).
./scripts/slurm/dispatch_all.sh --only order_probe

# 5. Watch.
squeue -u $USER -o "%.10i %.30j %.2t %.10M %R"

# 6. After all 32 jobs finish:
python -m interpretability.pipeline.order_probe_analyze --variants biased neutral
python -m interpretability.make_figures   # picks up figure_e automatically

# 7. Final audit + diff.
python scripts/audit_pipeline.py --save audits/after_order_probe.json
python scripts/audit_pipeline.py --compare audits/before_order_probe.json audits/after_order_probe.json
```

## Reading the output

`audit_pipeline.py` prints a headline section after Stage A' that gives mean
Jaccard and mean overlap@10 by `(variant, ordering_pair)`. The interesting
quantities:

- **`orig_vs_seedX`** — how stable is the rerank under re-ordering, vs the
  canonical SERP-position input?
- **`seed42_vs_seed123`** — noise-floor: two arbitrary random orderings.
- **Δ across variants** — does the neutral prompt produce more order-stable
  rankings than the biased one? (`figure_e_order_overlap_biased_vs_neutral.png`)

Rough interpretation guide for **mean overlap@10** between `orig` and a
shuffle:

| Mean overlap@10 | Interpretation                                                         |
|-----------------|------------------------------------------------------------------------|
| ≥ 0.8           | LLM is largely doing real re-ranking; small order-anchoring effect.    |
| 0.5 – 0.8       | Substantial order anchoring, but real preferences still dominate.      |
| ≤ 0.4           | Order-anchoring is a major confound; flag in the paper.                |

Compare to `seed42_vs_seed123` to subtract sampling/permutation noise — if
that pair is around 0.85 too, then "order anchoring" is the wrong story; the
LLM just has stable preferences modulo low temperature noise.

## Caveats

- `temperature=0.1`, not 0. Self-consistency is not perfect. Use the
  shuffle-vs-shuffle pair to estimate the noise floor.
- The shuffle is per-keyword (RNG keyed on `(seed, keyword)`) so adding new
  keywords later does not perturb existing ones.
- The probe writes to its own dir (`data/order_probe/`); it does not touch
  Stages B/C/D outputs and does not get joined into `full_experiment_data_*.parquet`.
- DML coefficients are NOT recomputed on shuffled outputs — this is a probe of
  the rerank step in isolation.
