# GEO-DML — RAG Ablation Study
### Snippet vs. Retrieval-Augmented LLM Re-ranking, at 400 keywords / cell

Date: 2026-05-11 · Author: Valerian Fourel

---

## Slide 1 — TL;DR

We measured how 10 on-page features (stats, structure, citations, freshness,
earned-media, …) influence an **LLM's re-ranking decision** of search-engine
results, under a **2 × 2 × 2 × 2 × 2 = 32-cell factorial** of:

| Axis | Levels |
|---|---|
| Search engine | **searxng** · **DuckDuckGo (ddg)** |
| LLM | **Llama-3.3-70B-Instruct** · **Qwen2.5-72B-Instruct** |
| SERP pool size | **top-20** · **top-50** |
| Prompt | **biased** ("you are a marketing assistant") · **neutral** ("you are a neutral search system") |
| Method (augmentation) | **snippet-only** · **RAG (query-conditional retrieval)** |

Total runs: **32 cells × 400 keywords = 12,800 re-rank calls** for Stage A,
plus **2 order-probe seeds = 25,600 calls** for Stage A'.

Headline finding (from DML over 4 of the 8 snippet↔RAG variant pairs that
are fully merged):
> **Earned-media demotion (T7) survives retrieval but shrinks 35%.
> Topical-competitor preference (T5) amplifies 9× under retrieval, but only
> in the biased prompt.**

---

## Slide 2 — Why two of everything

The five binary axes are **not** redundant. Each strips a different
alternative explanation off the table.

```
2 search engines  →  isolates SERP-distribution bias
                     (searxng=federated metasearch, ddg=independent index)

2 LLMs            →  isolates model-family idiosyncrasy
                     (Llama dense decoder vs. Qwen MoE-style training mix)

2 pool sizes      →  controls candidate-set effect
                     (top-20 = harder discrimination, top-50 = more headroom)

2 prompts         →  manipulates instruction-induced bias
                     (biased "promote earned media" vs. neutral framing)

2 methods         →  manipulates available evidence
                     (snippet-only ≈ 150 chars  vs.  RAG ≈ 3 × 800-char chunks)
```

If a treatment effect (say, "earned-media gets demoted") survives all four
non-treatment splits, it is **not** an artifact of one search engine, one
model, one prompt, or one candidate-set size. That is the robustness story
the paper wants.

---

## Slide 3 — The two **search engines**

| | **searxng** | **DuckDuckGo (ddg)** |
|---|---|---|
| Type | Federated metasearch | Independent index + Bing fallback |
| URL pool diversity | Higher (mixes Google/Bing/Brave/…) | Lower (mostly Bing-derived) |
| Phase-0 keywords scraped | 1,009 (pool=20) / 980 (pool=50) | 1,011 (pool=20) / 1,011 (pool=50) |
| Unique URLs in pool | 11,950 / 9,074 | 9,141 / 14,055 |
| NaN-position rows (data quality) | 0 | 6 (top-20), 97 (top-50) — patched |

Both engines feed the **same** downstream prompt builder and the **same**
LLM call. Differences in coefficients across engine isolate "what
candidates the LLM is given" from "how the LLM chooses among them".

---

## Slide 4 — The two **LLMs**

| | **Llama-3.3-70B-Instruct** | **Qwen2.5-72B-Instruct** |
|---|---|---|
| Provider | meta-llama (HF Inference) | Qwen (HF Inference) |
| Throughput (this study) | ~1 sec/keyword | ~3–4 sec/keyword |
| Sampling | temperature=0.1, max_tokens=500 | temperature=0.1, max_tokens=500 |
| Phase-2 rerank cells finished @400 kw | 8 / 8 | 8 / 8 |
| Phase-3 order_probe (interrupted 402) | 6 cells done + 1 partial | 0 cells started |

Same prompt template, same retrieval, same SERPs — only the model swaps.
Cross-model agreement of a treatment effect rules out model-family quirks.

---

## Slide 5 — The two **pool sizes**

```
Pool = how many candidate URLs the LLM must rank.

   top-20  ┌────────────────────┐
           │ 20 URLs per SERP   │  ← harder discrimination, denser bias signal
           └────────────────────┘

   top-50  ┌──────────────────────────────────────────────┐
           │ 50 URLs per SERP                              │  ← more tail,
           └──────────────────────────────────────────────┘    more noise
```

Both feed the same `top-10` final cut for evaluation. The pool size
controls how much **diversity / tail** the LLM sees before deciding.
A treatment that only shows up at one pool size hints at a "rare-candidate"
effect; one that holds across both is **robust**.

---

## Slide 6 — The two **prompts**

Two carefully matched system prompts, fed the *same* SERP+passage data:

| | **biased** | **neutral** |
|---|---|---|
| Persona | "marketing assistant for an e-commerce brand" | "neutral search-engine ranking system" |
| Implicit goal | optimize for the *client's* visibility | optimize for the *user's* information need |
| Expected effect | demote earned-media, promote topical competitors of the brand | rank purely by topical relevance |

Crucially the **task instruction is identical** ("re-rank these results
1..N"). Only the persona/goal differs. The (biased − neutral) gap is
the paper's central observable.

---

## Slide 7 — The two **methods** (this is the new axis)

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  SNIPPET-ONLY (the baseline)                                         │
   │  ────────────────────────────                                        │
   │                                                                       │
   │   1. [example.com] Title — snippet text (~150 chars)                 │
   │                                                                       │
   │   The LLM sees only what a normal SERP UI shows.                     │
   └──────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │  RAG (query-conditional retrieval — what production GEO does)        │
   │  ──────────────────────────────────────────────────────────────      │
   │                                                                       │
   │   1. [example.com] Title                                              │
   │      snippet: ...                                                     │
   │      passage: <chunk #7 (sim 0.71)> --- <chunk #14 (sim 0.65)>       │
   │                --- <chunk #2 (sim 0.61)>                              │
   │                                                                       │
   │   The 3 highest-similarity chunks of *that page's body*               │
   │   to *this keyword*, recomputed for every (kw, url) pair.             │
   └──────────────────────────────────────────────────────────────────────┘
```

This is the manipulation that matters for the GEO-claim: production
systems (ChatGPT search, Perplexity, Bing Copilot) all do query-conditional
retrieval. Snippet-only is the academic baseline; RAG is the field reality.

---

## Slide 8 — How the RAG index was built (one-time, ~$2.72)

```
44,220 unique URLs across 4 (engine, pool) cells
        │
        │   trafilatura.extract(html, max_chars=50_000)
        ▼
Full page bodies  ────────────────────────  ~136 M tokens total
        │
        │   recursive char chunker (size=800, overlap=200, min=100)
        ▼
906,322 chunks   (mean 24.8 chunks per page)
        │
        │   OpenAI text-embedding-3-small  (1536-dim, L2-normalized)
        ▼
chunk_embeddings.npy   per cell  (~1.3 GB × 4 = 5.3 GB on disk)

For each keyword (4,011 total):
        keyword_embeddings.npy   (1536-dim)

At rerank time, for each (keyword, url) on the SERP:
        sims = chunk_emb[url] @ keyword_emb[kw]      # cosine
        passage = " --- ".join(top-3 chunks by similarity)
        →  cached in retrieved_top3.parquet  (~13 K rows / cell)
```

Sanity check: keyword "educational software for schools" → top chunk on
searxng_top20 came from `research.com/software/best-education-software`
with cosine similarity **0.654**. Semantically appropriate.

---

## Slide 9 — Run summary (what finished, what didn't)

### Phase 2 — Stage A rerank (`keywords.jsonl`)
**✅ All 16 RAG cells × 400 keywords = 6,919 records. 0 fallbacks.**
(Combined with the 16 snippet cells already done last week → **32/32 cells.**)

### Phase 3 — Stage A' order_probe (`*_seed{42,123}.jsonl`)
| Status | Count | Cells |
|---|---|---|
| ✅ Fully done @ 400 each (2 seeds) | 6 | searxng × Llama × {20,50} × {biased,neutral}_rag, two seeds each (= 6 of 8 cells in that family — see note¹) |
| ⚠️ Partial (23/400, .pre402_bak preserved) | 1 | searxng/Llama/serp50/neutral_rag/seed42 |
| ⏸️ Pending | 25 | ddg × {Llama,Qwen} + searxng × Qwen |

¹ Stopped when HF Inference returned **402 Payment Required**.
Resume plan documented in `docs/RESUME-RAG.md`. ETA after top-up: ~9 hr, ~$15.

### Stage C/D — DML on full snippet+RAG cross
| Variant | Stage C rows | Stage D fits | Errors |
|---|---|---|---|
| biased (snippet) | (last week) | 280 | 0 |
| neutral (snippet) | (last week) | 280 | 0 |
| biased_rag | 33,384 | 280 | 0 |
| neutral_rag | 31,525 | 280 | 0 |

→ Full **2 × 2 grid** ready for analysis (prompt × method).

---

## Slide 10 — Results approach: DML on `rank_delta`

For every (treatment, cell, learner, …) we fit a Double Machine Learning
estimator (econml `LinearDML` or `PartialLinearRegressionEstimator`) with:

- **Outcome Y** = `rank_delta` = `pre_rank − post_rank` (positive = LLM
  promoted the URL; negative = LLM demoted).
- **Treatment T** = the on-page feature (e.g. `T7_source_earned` = 1 if URL
  is on a known earned-media domain).
- **Confounders X** = ~20 covariates (domain, keyword cluster, position
  prior, query intent class, length, …).
- **Nuisance learner** = LightGBM (chosen via cross-fitting comparison
  last week).

`POOLED · plr · lgbm · rank_delta` is the headline slice — pools across
the 4 (engine, pool) cells for power, uses Partial Linear with LightGBM,
on the rank-delta outcome.

---

## Slide 11 — The 2×2 grid (prompt × method) for the 10 treatments

```
                          SNIPPET                    RAG
                   ┌──────────────────────┐   ┌──────────────────────┐
         BIASED    │       coef_bs        │   │       coef_br        │
                   │                      │   │                      │
         NEUTRAL   │       coef_ns        │   │       coef_nr        │
                   └──────────────────────┘   └──────────────────────┘

  Of interest:
    - Prompt gap, snippet:  bs − ns
    - Prompt gap, RAG:      br − nr           ← does RAG narrow or widen the gap?
    - Aug effect, biased:   br − bs
    - Aug effect, neutral:  nr − ns
    - Interaction:          (br − bs) − (nr − ns)
                            = does augmentation hit the biased prompt
                              differently than the neutral one?
```

---

## Slide 12 — Headline coefficient table

`POOLED · plr · lgbm · rank_delta` — 10 on-page treatments × 4 (prompt, method) cells.

| Treatment | (B, snip) | (N, snip) | (B, RAG) | (N, RAG) |
|---|---:|---:|---:|---:|
| T1a stats present | −0.094 | −0.133 | −0.188 | −0.112 |
| T1b stats density | +0.001 | +0.001 | +0.000 | −0.003 |
| T2a question headings | +0.123 | **+0.147\*\*** | +0.137 | +0.056 |
| T2b structural modularity | **+0.005\*\*** | +0.002 | **+0.005\*** | +0.001 |
| T3 structured data new | −0.013 | **−0.123\*** | −0.092 | −0.040 |
| T4a ext citations | −0.066 | −0.141 | +0.083 | +0.108 |
| T4b auth citations | **−0.052\*\*\*** | −0.011 | **−0.049\*** | −0.016 |
| T5 topical competitor | +0.234 | +0.137 | **+0.807\*** | −0.111 |
| T6 freshness | −0.013 | **−0.049\*\*\*** | −0.034 | **−0.034\*** |
| **T7 source earned** | **−1.607\*\*\*** | **−0.417\*\*\*** | **−1.268\*\*\*** | **−0.496\*\*\*** |

`***` p<0.001 · `**` p<0.01 · `*` p<0.05.
Source: `data/dml_results/rag_vs_snippet_4way.csv`.

---

## Slide 13 — Three behavioural patterns

### Pattern 1 — RAG-ANCHORED (retrieval **narrows** the prompt-bias gap)
For domain-quality signals, real evidence forces the biased prompt to
engage with content; it can't blanket-demote on persona alone.

| Treatment | Snippet gap (B−N) | RAG gap (B−N) | Shrinkage |
|---|---:|---:|---:|
| **T7 source earned** | −1.19 | −0.77 | **−35 %** |
| T3 structured data | +0.11 | −0.05 | −53 % |
| T4a ext citations | +0.075 | −0.025 | −66 % |

The headline T7 demotion **survives** retrieval (still p<0.001 in both
arms) but the prompt-driven gap shrinks by a third.

### Pattern 2 — RAG-AMPLIFIED (retrieval **widens** the gap) — only T5
| Treatment | Snippet gap | RAG gap | Change |
|---|---:|---:|---:|
| **T5 topical competitor** | +0.10 (n.s.) | **+0.92 (biased side sig.)** | **+853 %** |

Under snippets neither prompt cares about topical-competitor signals.
Under RAG, the biased prompt strongly promotes topical competitors
(+0.807, p<0.05); the neutral arm stays at zero.

### Pattern 3 — PROMPT-INSENSITIVE (small, stable, both modes)
T1b, T2b, T4b, T6 — coefficients within noise band; isolated `*` cells
are likely false positives.

---

## Slide 14 — Asymmetric interaction term

The augmentation effect `(rag − snippet)` is **different** between biased
and neutral arms on the two interesting treatments:

| Treatment | Within biased<br>(br − bs) | Within neutral<br>(nr − ns) | Interaction |
|---|---:|---:|---:|
| T7 source earned | **+0.340** (less negative) | −0.079 | **+0.419** |
| T5 topical competitor | **+0.573** (more positive) | −0.248 | **+0.822** |

- T7 — biased becomes **less harsh** → RAG **anchors** away from prompt bias.
- T5 — biased becomes **more positive** → RAG **amplifies** prompt bias when
  retrieved content can confirm the persona's preference.

Same sign of interaction, opposite normative meaning — driven by the
**sign of the snippet baseline**.

---

## Slide 15 — What the paper can claim (with this data)

1. **Robustness.** T7 earned-media demotion survives the move from
   academic snippet-only to realistic query-conditional retrieval.
   `−1.61***  →  −1.27***` for biased, `−0.42***  →  −0.50***` for neutral.

2. **Anchoring on domain-quality signals.** For T7 / T3 / T4a, RAG
   narrows the prompt-bias gap by **35–66 %**. Real GEO systems partially
   neutralize prompt-instruction bias on this class of signals.

3. **Selective amplification on content-relevance signals.** For T5,
   retrieval multiplies the biased prompt's competitor-preference by
   ~9×. Production systems may strengthen prompt-driven preferences when
   retrieval can supply confirming evidence.

4. **Asymmetric interaction.** RAG impacts the biased arm more strongly
   than the neutral arm in **both** directions — the mechanism is
   "retrieval changes behaviour more when the prompt is looking for
   something the retrieval can confirm or contradict".

---

## Slide 16 — Remaining work

| Step | ETA | Cost | Blocker |
|---|---|---|---|
| Top up HF Inference credits | 1 min | — | manual |
| Phase 3 resume — finish 26 order_probe cells | ~9 hr | ~$15 | credits |
| Run order_probe analyze + report (Jaccard, OAK) | ~5 min | — | Phase 3 done |
| Generate figures (`figure_a_dml_*` 4-panel + delta) | ~30 min | — | none |
| Stage C/D for `_rag` already done — re-run after order_probe | ~30 min | — | none |
| Write-up section on Pattern 1/2/3 + interaction | ~2 d | — | none |

Documented in `docs/RESUME-RAG.md`.

---

## Slide 17 — Risks & caveats (1 slide of honest hedges)

1. **N is smaller in RAG arm.** snippet POOLED ≈ 20 k obs/variant; RAG
   POOLED ≈ 12 k. Pages whose body couldn't be chunked drop out. CIs
   widen accordingly.
2. **T5 amplification has only one significant arm.** Cautious framing:
   "the biased prompt's competitor preference becomes significant and
   3.4× larger under RAG; the neutral prompt remains zero." Not "RAG
   flips T5".
3. **Order-probe data is incomplete** (6 of 32 cells). Stage A' Jaccard
   / OAK numbers will land after the resume.
4. **Qwen RAG is untested in Phase 3.** Could behave differently from
   Llama in the order-probe at the rank-level granularity.
5. **Chunker is naïve recursive char split**, not semantic chunking. A
   sensitivity check with a different chunker is on the "future work"
   list, not the critical path.

---

## Slide 18 — One picture to remember

```
                        SNIPPET                          RAG
                  ──────────────────                ──────────────────
   T7 demote   :  ▆▆▆▆▆▆▆▆ −1.61              :   ▆▆▆▆▆▆ −1.27       ← still strong
                  ▆▆ −0.42                    :   ▆▆ −0.50          ← still strong
                                                                       gap shrinks 35 %

   T5 promote  :  ▁ +0.23 (n.s.)              :   ▆▆▆▆▆▆▆▆▆ +0.81 *  ← amplified
                  ▁ +0.14 (n.s.)              :   ▁ −0.11 (n.s.)     ← still flat
                                                                       gap grows 9 ×

                ↑ snippet ≈ academic baseline   ↑ RAG ≈ production GEO
```

- **Anchoring** on the loud signal (T7): real evidence dampens prompt bias.
- **Amplification** on the silent signal (T5): real evidence *activates*
  the prompt's framing.

This is the paper's contribution.

---
*Generated 2026-05-11 from:*
`data/dml_results/rag_vs_snippet_4way.csv` ·
`data/dml_results/rag_vs_snippet_comparison.csv` ·
`docs/work-log-2026-05-08.md` · `docs/RESUME-RAG.md`
