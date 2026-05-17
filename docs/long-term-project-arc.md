# Long-term project arc

A grounded, end-to-end view of what this repo is for. Reads three layers
deep — what we are building short-term (EMNLP 2026), what we are
*characterizing* in the process, and what those characterizations are
ultimately *for* (a distilled, locally-deployable specialized search
reranker).

Sources: README.md, work logs 2026-04-28/29/30, the actual modules under
`interpretability/` and `interpretability/pipeline/`. File:line citations
are inline so claims can be checked against the code.

---

## 0. TL;DR

We are running a controlled experiment on how large LLMs (Llama-3.3-70B,
Qwen2.5-72B) rerank search results. The experiment lives at the
intersection of three axes — **what the LLM is told** (biased vs neutral
prompt), **how much information it gets per result** (snippet-only vs
passage-augmented), and **which information regime is used to validate the
finding** (a causal-effect estimate via DML, plus three independent
mechanistic interpretability tests).

Either of two outcomes is a paper:

- **Exploit-style finding.** The LLM applies one of the on-page
  treatments (T1–T7) in a way that publishers can game, or fails in a way
  that adversaries can exploit (e.g. position-bias amplification under the
  biased prompt's "exclusion list" instruction, snippet-only fragility).
- **Characterization-style finding.** The LLM behaves *robustly* across
  axes; we now have a quantitative map of where, when, and how strongly
  each on-page treatment moves rank, plus the failure modes (order
  sensitivity, candidate-set starvation, prompt-instruction confounds).

Both outcomes feed the same downstream goal: enough mechanistic
understanding of the teacher's reranking skill to **distill it into a
small student model that can be deployed locally** as a specialized
search reranker. The current paper is step 1; the distilled model is
step 2; local search inference is step 3.

---

## 1. The four-stage arc

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 1 — Causal + mechanistic study (this repo, EMNLP 2026)            │
│   DML coefficients + ablation + saliency + probing + order probe        │
│   over (model × engine × pool × prompt-variant × content-mode) cells.   │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 2 — Findings consolidate into ONE of                              │
│   (a) an exploit story — specific, gameable behaviors of the LLM        │
│   (b) a characterization story — robust behaviors + measured pitfalls   │
│   Either outcome is a defensible paper.                                 │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 3 — Distillation spec sheet                                       │
│   The DML coefficients, saliency atlases, probing per-layer maps, and   │
│   order-stability profiles BECOME the supervision targets and the       │
│   inductive biases for a smaller student model.                         │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 4 — Local specialized reranker                                    │
│   Distilled student (likely 1B–7B) that does the rerank skill, runs on  │
│   commodity hardware, no API dependency. Ships as the "GEO-aware local  │
│   search" model.                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

The repo is currently working through Stage 1. Stages 2–4 are not built
yet, but the artifacts produced in Stage 1 are designed so they feed
directly into Stage 3.

---

## 2. What the current pipeline produces, and why each piece matters for the arc

The audit (`scripts/audit_pipeline.py`) tracks six stages. Each is named
by its role in the *current paper*; here we tag each with its role in
the *long-term arc*.

### 2.1 Stage A — Rerank (`interpretability/pipeline/rerank.py`)

**What it does.** For each cell `(model, engine, pool)` and each
prompt variant, takes the cached SERP, builds the rerank prompt
(`prompts.build_rerank_prompt_with_spans`,
`interpretability/pipeline/prompts.py:110`), calls the LLM once per
keyword, and writes a JSONL with the new top-N ranking plus the per-result
character spans. This is the canonical record of "how the teacher LLM
actually ranks the SERP."

**Role in the arc.** This *is* the teacher's behavior. Stage 4's
distilled student will train, at minimum, on `(prompt, ranking)` pairs
extracted from these JSONLs. Every other stage is either explaining or
validating these rankings.

The variant axis is currently 4-wide (work log 2026-04-30 §2):

| variant            | header / footer | per-result block | added work_log |
|--------------------|-----------------|--------------------------------------------|---------|
| `biased`           | "rerank software product domains; exclude review aggregators, blogs, news…" | `{n}. [{domain}] {title} — {snippet[:150]}` | original |
| `neutral`          | exclusion list dropped, "software product" framing dropped | same one-line block | 04-28 |
| `biased_passage`   | biased header/footer | multi-line block: snippet + `passage[:800]` from trafilatura | 04-30 |
| `neutral_passage`  | neutral header/footer | same multi-line block as biased_passage | 04-30 |

**Why all four.** The variant axis isolates two confounds simultaneously.

- **biased vs neutral** addresses the prompt-instruction confound
  identified on 04-28 §3: the original biased prompt explicitly tells the
  LLM "exclude review aggregators, directories, Wikipedia, news, blogs,
  forums, YouTube" — which is exactly the category T7 (source_earned)
  covers. So a measured T7 demoter effect is partly tautological under
  the biased prompt. The neutral arm strips that instruction and
  re-measures.
- **snippet vs passage** addresses the information-richness confound
  identified on 04-30 §2: production GEO systems (Perplexity Sonar,
  Google AI Overviews, Bing Copilot, ChatGPT Search, the original GEO
  benchmark) feed the LLM passage-level body text, not 150-char SERP
  snippets. If the prompt-instruction effect goes away once the model
  has body text to anchor on, the snippet-only finding doesn't
  generalize to production. If it persists, the effect is robust to
  information regime.

The four-cell bracket is the empirical hook of the paper. Per work log
2026-04-30 §2: "if `biased_passage ≈ neutral_passage` while
`biased ≠ neutral`, the prompt-instruction effect is a snippet artifact
and doesn't generalize to production GEO — that's the more interesting
paper. If the gap persists across both information conditions,
prompt-instruction bias is robust to information richness, which is also
publishable."

### 2.2 Stage A' — Order probe (`interpretability/pipeline/order_probe.py`)

**What it does.** Reruns Stage A two more times per cell with the input
SERP shuffled under fixed seeds 42 and 123 (so two new orderings per
cell, plus the original). Then `order_probe_analyze.py` computes pairwise
Jaccard / overlap@K (K∈{3,5,10}) across the three orderings.
`order_probe_report.py` (added 04-30 §1) emits per-cell CSVs slicing the
overlap by `(variant, model, engine, pool)`.

**Role in the arc.** Tests whether the LLM is *ranking* or *anchoring*.
If a +0.8 rank-delta on T7 reflects judgment, the same set of items
should bubble up regardless of input order. If it just reflects "the
model copies SERP order with light noise," all DML coefficients
downstream are conditional on a fragile artifact.

The 04-30 §1 findings are the empirical output. K=10 mean overlap:

- **biased**: 0.69 (heavy variation across cells; concentrated in
  `searxng/pool=50` with Δoak ≈ -0.27 vs neutral)
- **neutral**: 0.82 (much more stable across cells)

Reading rule (work log 2026-04-30 §1): the prompt-induced position-bias
amplification is mostly a `searxng/pool=50` phenomenon. Many of the
worst-overlap keywords have `mean_oak ≈ 0.10` and `mean_jacc = 1.00` —
i.e. the LLM returned only one valid domain on both runs and it was the
same domain. This happens for queries like *"stress relief techniques"*
where the biased prompt's "software product" instruction has no eligible
candidates. None of these show up under neutral.

**Distillation implication.** A student trained on the biased-prompt
rankings would inherit that fragility. The order-probe tells the
distillation step which input modes need data augmentation
(permutation-invariance training) and which keyword classes need
candidate-set sanity gates.

### 2.3 Stage B — Features (`interpretability/pipeline/features.py`)

**What it does.** Variant-agnostic deterministic feature extraction over
each cell's HTML cache. Produces:

- The 10 NEW treatments (`treat_stats_present`, `treat_stats_density`,
  `treat_question_headings`, `treat_structural_modularity`,
  `treat_structured_data`, `treat_ext_citations_any`,
  `treat_auth_citations`, `treat_topical_comp`, `treat_freshness`,
  `treat_source_earned`) — see `config.py:87` `TREATMENTS_NEW`.
- 24 confounders (`config.py:133` `CONFOUNDERS`) — title/snippet
  similarities, length, brand recognition, BM25, HTTPS, Moz domain
  authority, DataForSEO keyword-difficulty / search-volume / CPC /
  intent-classes.
- Optional sentence embeddings for topical-competence (T5).

Per work log 2026-04-28 §4 (locked decisions): **deterministic-only**
extraction. Pure regex / BeautifulSoup / JSON-LD parsing. No LLM call
for feature scoring. This was a deliberate choice: an LLM-scored
treatment column would mix the very signal we are trying to isolate
(LLM judgment) into the inputs of the causal model.

**Role in the arc.** Stage B is the *features matrix* for DML. It
separates the on-page properties of each result (the X, Z) from the
LLM's behavior on it (the Y, T). Without this separation we cannot
identify the causal effects. In the distillation arc, these treatment
columns are also the *labels* for any contrastive supervision: a
counterfactual pair `(page, page_with_T_added)` needs T to be
deterministically identifiable to be a clean training signal.

### 2.4 Stage C — Merge (`interpretability/pipeline/merge.py`)

**What it does.** Pure-pandas join: SERP × LLM rerank × features ×
confounders → `data/main/full_experiment_data_{variant}.parquet`. One row
per `(keyword, url, model, engine, pool, variant)`. Carries `pre_rank`
(SERP), `post_rank` (LLM), `rank_delta = pre_rank - post_rank`, all
treatments, all confounders.

**Role in the arc.** This is the input to DML. It is also the natural
substrate for distillation training-set construction: every row is a
candidate (prompt-context, candidate-page, teacher-rank) tuple.

### 2.5 Stage D — DML (`interpretability/pipeline/dml.py`)

**What it does.** Runs DoubleML (PLR / IRM, with LightGBM and Random
Forest as nuisance learners) over a grid of `(subset × outcome ×
treatment × method × learner)`. Subsets are POOLED, by_engine,
by_model, by_pool, by_engine_model_pool. Outcomes are `rank_delta` and
`post_rank`. Treatments are the 10 new ones plus 4 code-extracted plus
4 LLM-scored (off by default).

Output is `dml_results_long_{variant}.parquet`: one row per fit,
columns `coef, se, t_stat, p_val, ci_lower, ci_upper, sig_stars`.

**Role in the arc.** This is *the causal contract*. The headline
coefficients from the original paper:

- T7 (source_earned): **coef = -1.7**, p<0.05 (strong demoter)
- T5 (topical_comp): **coef = +0.44** (strong promoter)
- T3 (structured_data_new): -0.14
- T2a (question_headings): +0.10
- T6 (freshness): -0.06
- T1b (stats_density): -0.02

These are the targets a distilled student should preserve. A student
that ranks correctly on average but does not respect these marginal
effects — e.g. ranks the right urls but fails to demote earned sources
— has lost the skill that DML measured.

The four-variant matrix means we will end up with four such tables,
and the *robustness* of each coefficient across variants tells us
which are mechanistically real and which are prompt artifacts.

### 2.6 Stage F — Mechanistic interpretability

Three independent methods, each producing one dimension of mechanistic
evidence about whether the DML coefficients are real *inside* the LLM
or just shadows of statistical patterns in the data.

#### 2.6.1 Ablation (`interpretability/ablation.py`)

**Method.** For each treatment, remove the treatment-relevant tokens
from the candidate's snippet (e.g. for T7, prepend "Official vendor
page:" to non-earned URLs and strip the press-coverage cue from earned
URLs; for T5, strip the keyword tokens from title/snippet; see
`ablation.py:100-116`). Re-rank. Measure `abl_delta = ablated_rank -
baseline_rank`.

**Reading rule.** If the DML coefficient says T7 is a -1.7 demoter and
ablation says removing the source-earned signal moves rank by -0.28
(full frame) or -0.02 (robust-winners frame), the methods agree on
direction.

**Frames.** `full` runs on all keywords. `robust_winners` restricts to
(keyword, url) pairs the LLM placed top-10 under both serp20 *and*
serp50 — same conditioning as the §4.1 DML headline (see
`interpretability/_robust_winners.py`). The robust-winners frame
isolates the position-shift effect from the selection effect.

#### 2.6.2 Saliency (`interpretability/saliency.py`)

**Method.** Gradient × input-embedding saliency on a 4-bit quantized
8B proxy (Llama-3.1-8B-Instruct). Run the rerank prompt forward, take
the gradient of the rank score w.r.t. the input embeddings. The
treatment's pre-computed character spans (`domain_span` for T7,
keyword spans for T5, etc. — built in
`prompts.build_rerank_prompt_with_spans`) tell us which token positions
are "treatment tokens"; saliency on those vs the rest is the ratio.

**Reading rule.** For a treatment the LLM actually attends to, the
saliency ratio (treatment-token mean / other-token mean) should exceed
1. If saliency is flat across treatment tokens, the DML coefficient is
"explained at distance" — it shows up in outputs without leaving a
gradient trail.

#### 2.6.3 Probing (`interpretability/probing.py`)

**Method.** For each transformer layer L of a local proxy model, train
a logistic regression probe on layer-L hidden states (pooled over the
treatment span) to predict the binarized treatment label. Reports
per-layer accuracy + ROC-AUC.

**Reading rule.** A treatment is *encoded* in the model when probe
accuracy >> chance. The layer at which it peaks tells you where in the
network it's being computed:
- T7 (source_earned) hypothesis: early-mid layers (structural / surface
  cues — "blog", "news" are tokens the LLM categorizes early).
- T5 (topical_comp) hypothesis: late layers (requires semantic
  understanding of keyword vs page topic).

Empirical (work log 2026-04-28 §2): T7 hits 99% accuracy at layer 39 of
the 8B proxy under the biased prompt. That is a representational fact
about the model — independent of the prompt phrasing — and one of the
findings most likely to survive the prompt-instruction critique.

#### 2.6.4 Weights (`interpretability/weight_analysis.py`)

**Method.** Per-layer weight-norm and head-norm analysis as a sanity
check; complements probing. Output is large (max_rows=81730 in the
audit) — the per-head, per-layer norm table.

**Role in the arc.** All four methods together produce the
**mechanistic atlas** that becomes the distillation spec: which layers
carry which features, which tokens drive which decisions, which
treatments survive ablation. A student model designer who has read the
atlas knows exactly which capabilities to preserve and which the
teacher gets right by accident.

---

## 3. The exploit-vs-characterize duality

The user framed this as "either we find an exploit in the treatment, or
we have an idea of how an LLM acts for search." These are not
mutually exclusive — they are the same finding viewed from two angles.

### 3.1 What "exploit" can mean here

Concrete exploits the current pipeline can surface:

1. **Prompt-induced exclusion bias** (the 04-28 §3 finding). The biased
   prompt's "exclude review aggregators, Wikipedia, news, blogs" line
   makes T7 a tautological demoter. Adversarial publishers learn:
   structure your page so it parses as "vendor product" rather than
   "earned media" and you escape the demotion regardless of content
   quality. This is already an actionable GEO playbook.

2. **Biased-prompt position-bias amplification** (the 04-30 §1
   finding). `searxng/pool=50` under the biased prompt has Δoak ≈
   -0.27 vs neutral; the LLM is much more sensitive to input order
   when given exclusionary instructions. Adversaries who can influence
   *which* SERP results come first (via SearXNG / DDG / Google SEO)
   get amplified leverage when the downstream LLM has a biased
   reranking prompt.

3. **Candidate-set starvation** (the 04-30 §1 finding). For queries
   like "stress relief techniques" the biased prompt's "software
   product" instruction leaves the LLM with one or zero eligible
   candidates. That is itself an exploit surface: a publisher with the
   single eligible page on a starved query gets locked in across
   permutations and prompt variants.

4. **Snippet-vs-passage modality dependence** (the 04-30 §2 prediction
   — to be confirmed). If `compare_variants.py --pair-grid` shows
   Kendall's τ ≤ 0.4 between snippet and passage rerankings, then the
   information regime is doing meaningful work and snippet-only
   listwise rerankers are not safe to deploy in production GEO.

### 3.2 What "characterization" can mean here

If the bracket experiment shows the findings are *robust* across all
four variants, the paper is no longer "we found an exploit" but "we
have the first quantitative map of how a 70B-class LLM actually does
search reranking." That is also publishable, and is in fact the
*better* setup for the distillation work that follows: it means the
skill is real, repeatable, and worth distilling.

The current evidence (per work logs) leans toward **partial robustness
with specific exploits**:

- T7 probing at 99% accuracy is robust (representational fact).
- T7 DML coefficient is partly tautological under biased.
- Order stability is dramatically worse under biased
  (`searxng/pool=50` especially).
- Stage F results pending for neutral and the two passage variants.

The distinction matters less than it first looks: in either case, we
end up with the same artifact — a mechanistic atlas that says where the
skill lives, where the failure modes are, and which inputs trigger
them. That atlas is what makes Stage 3 (distillation) tractable.

---

## 4. Distillation spec sheet — three layers of supervision

This section is the bridge from "we have a paper" (Stage 2) to "we
have a deployable model" (Stage 4). It is *not implemented yet*; it
documents the shape of the future training pipeline given what the
current pipeline produces.

### 4.1 Layer 1 — Behavioral imitation (cheap, weak)

**Signal.** The Stage A `keywords.jsonl` files. For each of ~5,300
keywords across 8 cells × 4 variants, we have `(prompt, top-N
ranking)` from a 70B teacher. After Stage A is fully complete: ~21,000
labeled rerank examples per variant pair.

**Distillation loss.** Standard rerank distillation: train the
student to emit the same ranking, optionally weighted by the position
of each domain. This is exactly what monoT5 / RankT5 / RankZephyr /
RankLLaMA already do.

**Limitation.** The student inherits whatever spurious heuristics the
teacher has — including, by construction, the prompt-instruction
artifact and the order-anchoring fragility. Layer 1 alone gives you a
faster reranker, not a *better* one.

### 4.2 Layer 2 — Causal target preservation (the differentiator)

**Signal.** The Stage D `dml_results_long_{variant}.parquet` rows.
Each row is a contract: "treatment T applied to a page changes its
expected `rank_delta` by β, with confidence interval CI".

**Distillation loss.** Construct counterfactual training pairs from
the Stage C parquet. For each `(keyword, url)` row with treatment
T = 1, find a matched row with T = 0 (same keyword, similar
confounders). Train the student so that

```
E[student_rank(page_T1) - student_rank(page_T0)] ≈ β_T
```

with a margin loss penalizing departures > 2σ. This is *causal
distillation*: instead of imitating the teacher's outputs, imitate the
teacher's marginal causal effects.

**Why it matters.** Two students with the same overall rerank
accuracy can have wildly different sensitivities to specific on-page
properties. The DML coefficients tell us which sensitivities are
*real* and *causal*; a student trained to preserve them is robust to
distribution shift in a way pure imitation cannot guarantee.

**Why we needed Stage F to do this honestly.** A DML coefficient
alone is just a number; we cannot tell whether to teach the student
to preserve it without knowing whether it reflects mechanistic
behavior or a prompt artifact. Stage F is the filter: only
coefficients that survive ablation + saliency + probing are loaded
into the student's spec sheet.

### 4.3 Layer 3 — Pitfall avoidance (inductive biases)

**Signal.** Order-probe overlap profiles, biased-vs-neutral coefficient
deltas, snippet-vs-passage τ values. None of these are training
*targets* — they are training *constraints*.

**Translations to architectural / data choices.**

- **Order stability.** The order-probe says biased-prompt rerankers
  are permutation-fragile. Counter-measure: train the student with
  permutation augmentation — for each (prompt, ranking) example,
  generate several permuted variants and require the same output.
  Expected outcome: a student with mean_oak > 0.85 across all cells.
- **Prompt-instruction robustness.** The biased-vs-neutral delta says
  certain treatment effects are prompt-conditional. Counter-measure:
  train the student on the union of all four variants, not just
  biased; explicitly include neutral and passage examples so the
  student learns content-grounded ranking rather than
  instruction-conditional ranking.
- **Modality robustness.** If snippet-vs-passage τ is low, the
  student needs to be trained on both modalities, with the modality
  available as a feature (or as the actual input format) — not on
  snippet-only data with passage as a hoped-for generalization.
- **Candidate-set sanity.** The "stress relief techniques" failure
  case says the student needs an explicit *abstain* path when the
  candidate pool has no eligible domains. This is a model-card
  decision (output format must allow `[]`) and a training-data choice
  (include the candidate-starved keywords with `[]` as the gold
  label).

---

## 5. The local specialized reranker (Stage 4)

### 5.1 Prior art that already exists

- **monoT5 / RankT5** (Nogueira et al.): T5-base/large fine-tuned for
  pointwise / listwise reranking. Industry baseline.
- **RankZephyr** (Pradeep et al., 2023): 7B Zephyr distilled from a
  GPT-4 listwise reranker. Currently best-published open small model
  for listwise rerank.
- **RankLLaMA** (Ma et al., 2023): LLaMA-2 7B / 13B fine-tuned for
  passage reranking with a pointwise loss.
- **RankVicuna**: similar lineage.

All of these are Layer-1-only distillation. None of them use causal-
effect supervision, none of them are conditioned on a mechanistic
atlas, none of them are explicitly trained to be order-stable or
prompt-robust.

### 5.2 What this project adds

The differentiator is not "we built another small reranker." It is
"we built a small reranker whose training signal includes the validated
causal effects of the teacher, the order-stability constraint, and the
prompt-robustness constraint."

If the EMNLP paper lands, the distillation work writes itself: the
paper *is* the spec sheet. A NeurIPS / ACL follow-up titled something
like "Causally-distilled rerankers: small models that preserve the
treatment effects of frontier LLMs" is the natural Stage 3 → Stage 4
output.

### 5.3 Architecture choices and gotchas

- **Student size.** 1B–7B is the realistic band. Sub-1B will struggle
  with passage-mode context length (10 results × ~1030 chars ≈ ~2.5K
  tokens of input). 7B (e.g. Qwen2.5-7B, Llama-3.1-8B) is comfortable
  on a single 24GB GPU at fp16.
- **Local-first.** No HF Inference dependency. Quantize to 4-bit
  (bitsandbytes / GPTQ) for commodity GPUs. The repo already runs the
  saliency and probing analyses on 4-bit 8B locally
  (`README.md:96-105`); the same path applies to inference.
- **Counterfactual training pair construction.** This is the one piece
  that does not exist yet. Needs:
  - A treatment-application pipeline (synthetic perturbation: add
    schema markup / strip earned-media cues / add stats / etc.) so
    that for each `(keyword, url)` row we can generate
    `(keyword, url_with_T)` and re-feed it to the teacher.
  - Or: matching-based pseudo-counterfactuals from the Stage C parquet
    (find row pairs that differ only in T, use the teacher's actual
    rank gap as the supervision target).
- **Eval.** Beyond standard rerank metrics (NDCG, MRR), report the
  same Stage F metrics on the student: does it have the right
  saliency ratios? does its probing curve match the teacher's? does
  it have the same DML coefficients on a held-out test set? These
  are the new evaluation axes the EMNLP paper introduces.

### 5.4 The end product

A specialized model checkpoint + an inference harness that takes a
search query + a list of result URLs/snippets/passages and emits a
ranking. Latency target: <500ms / 10-result rerank on a single
consumer GPU. Memory target: fits in 24GB VRAM at 4-bit. License: open
weights, since the teachers are open (Llama-3.3, Qwen2.5).

The deployment vision is "GEO-aware local search reranker" — a
component that sits behind any local search frontend (Perplexica,
Open-WebUI search plugins, custom RAG pipelines) and replaces an API
call to a frontier reranker with a local one whose behavior is
characterized, robust, and causally validated.

---

## 6. What's complete now, what's missing for Stage 4

### 6.1 Stage 1 — current paper (where we are)

Per `python scripts/audit_pipeline.py` as of 2026-05-01:

| Component                      | Status                  |
|--------------------------------|--------------------------|
| Inputs (SERP + HTML)           | Complete                |
| Stage A (rerank)               | 24/32 (ddg passage variants pending) |
| Stage A' (order probe)         | 48/64                   |
| Stage A' analysis              | 96k rows ✓              |
| Stage B (features)             | 0/4 — bottleneck        |
| Stage C (merge)                | 0/4                     |
| Stage D (DML headline table)   | 0/4                     |
| Stage F: ablation              | 48/48 ✓                 |
| Stage F: saliency              | 16/16 ✓                 |
| Stage F: probing               | 2/8 (6 cells missing)   |
| Stage F: weights               | 8/8 ✓                   |

The bottleneck is Stage B → C → D. Once features lands, the headline
coefficient table can populate, the figures can be made, and the
paper draft can be assembled.

### 6.2 Stage 2 — findings consolidated

Drafted in the work logs but pending the final DML run:

- Confirmed: T7 probing is decodable from layer 39 (representational fact).
- Confirmed: order stability is much worse under biased
  (`searxng/pool=50` especially).
- Pending: T7/T5/T2a coefficient deltas across all four variants
  (the bracket-experiment headline figure).
- Pending: passage-arm vs snippet-arm Kendall's τ (the modality test).
- Pending: per-treatment robustness map (which coefficients survive
  all four variants, which collapse).

### 6.3 Stage 3 — distillation spec sheet

Not built. To be built post-EMNLP. Inputs from Stages 1+2:

- Per-treatment causal coefficient + 4-variant robustness profile.
- Per-treatment per-layer probing map.
- Per-treatment saliency atlas (which token positions matter).
- Order-stability profile per cell.
- Failure-mode catalog (candidate-starved keywords, etc.).

### 6.4 Stage 4 — local reranker

Not built. Architecture sketch in §5.3 above. Key prerequisite is the
counterfactual training-pair pipeline; this is a new module, not
adapted from the current code.

---

## 7. Honest open questions and risks

### 7.1 Risks to the EMNLP submission

- **The bracket may show no effect.** If
  `biased ≈ neutral ≈ biased_passage ≈ neutral_passage`, the paper
  loses its sharpest result. Fallback story: "we built a robust
  measurement methodology and confirmed prior findings replicate" —
  publishable but less interesting.
- **Stage F probing for neutral / passage variants is unfinished**
  (work log 2026-04-30 §4 / today's audit). Without those, the paper's
  Figure C is biased-only and the mechanistic story is one-armed.
- **The DML reproduction gate** (work log 2026-04-28 §6 step 2). The
  port has not yet been validated against the paper's existing
  `dml_results_long.parquet` (the "biased reproduction gate"). If
  `max |Δcoef| > 0.01`, the port has a bug and all four variants need
  re-running.

### 7.2 Risks to the distillation arc

- **Counterfactual training pairs are expensive.** Synthesizing
  `(page, page_with_T)` pairs requires either real perturbation (HTML
  rewriting + recrawling — non-trivial) or pseudo-counterfactual
  matching from the existing parquet (cheap but introduces matching
  bias). Either way, the Stage 3 dataset is not free.
- **Causal distillation is unproven at scale.** No published work
  trains a small reranker against the marginal causal effects of a
  large one. Stage 3 is a research bet, not a known engineering
  exercise.
- **Two teachers.** The current pipeline produces rankings from both
  Llama-3.3-70B and Qwen2.5-72B. The student can be distilled from
  either, both, or an ensemble. Choice has implications: one teacher's
  prompt-instruction artifact might survive into the student even if
  the other teacher's didn't.
- **Modality choice for the deployable student.** Snippet-only is
  cheaper to serve; passage-augmented is closer to production GEO. If
  the bracket shows passage augmentation matters, the deployable
  student must include a passage-extraction step at inference time
  (trafilatura on the candidate page; the same path
  `interpretability/utils.py:197` `extract_passage` already
  implements).

### 7.3 Risks to the local-deployment story

- **Quantization × specialization interaction.** A 4-bit quantized
  reranker may lose some of the mechanistic features the probing atlas
  identified. Stage 4 needs a quantization-aware checkpoint of the
  Stage F validation methodology.
- **Domain drift.** The current dataset is fixed (cached SERPs from
  some past crawl). A deployed local reranker faces a moving SERP
  distribution. Without periodic teacher-validation runs, the student
  drifts away from the causal contract.

---

## 8. Cross-references

- `README.md` — short user-facing description and quickstart.
- `docs/work-log-2026-04-28.md` — initial bias finding, port of
  upstream pipeline into this repo, neutral prompt design.
- `docs/work-log-2026-04-29.md` — order-probe design, variant-aware
  Stage F, audit infrastructure.
- `docs/work-log-2026-04-30.md` — passage-augmented arm, per-cell
  order-probe report, cluster-ops gotchas.
- `docs/pipeline-port.md` — operator runbook for the A→D chain.
- `docs/order-probe.md` — operator runbook + interpretation thresholds
  for the order-stability experiment.
- `docs/next-steps-prompt.md` — original locked decisions for the port.

Code anchor points referenced in this document:

- `interpretability/pipeline/prompts.py:110` — variant-aware prompt builder.
- `interpretability/pipeline/rerank.py:242` — passage-map extraction.
- `interpretability/pipeline/order_probe.py` — Stage A' driver.
- `interpretability/pipeline/order_probe_analyze.py` — overlap analysis.
- `interpretability/pipeline/order_probe_report.py` — per-cell CSV slices.
- `interpretability/pipeline/features.py` — deterministic feature
  extraction.
- `interpretability/pipeline/dml.py` — DoubleML grid driver.
- `interpretability/pipeline/config.py:73-100` — treatment definitions.
- `interpretability/pipeline/config.py:133-160` — confounder list.
- `interpretability/utils.py:197` — `extract_passage` (trafilatura).
- `interpretability/ablation.py` — Stage F option 1.
- `interpretability/saliency.py` — Stage F option 2.
- `interpretability/probing.py` — Stage F option 3.
- `interpretability/weight_analysis.py` — Stage F weights.
- `interpretability/_robust_winners.py` — robust-winners frame helper.
- `scripts/audit_pipeline.py` — six-stage audit + Stage F section.
- `scripts/compare_variants.py` — pairwise variant diagnostics
  (Kendall τ, RBO@10, Jaccard@10).
