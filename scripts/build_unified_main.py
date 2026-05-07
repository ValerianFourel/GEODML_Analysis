#!/usr/bin/env python3
"""Build a single DML-ready parquet that unions all 4 variants and exposes
every axis-of-variation as an explicit column.

Output: $GEODML_DATA_ROOT/data/main/full_experiment_unified.parquet

Each row is one (keyword, url, model, engine, pool, prompt, passage_mode)
observation with all treatments, all confounders (HTML-derived,
keyword-level DataForSEO, Moz, BM25), and the outcome columns.

Axes (call them explicit columns rather than packed in run_id):
  axis_engine        ∈ {searxng, ddg}
  axis_model         ∈ {Llama-3.3-70B-Instruct, Qwen2.5-72B-Instruct}
  axis_pool          ∈ {20, 50}
  axis_prompt        ∈ {biased, neutral}              (SEO-framing vs neutral)
  axis_passage_mode  ∈ {snippet, passage}             (RAG off vs on)
  axis_top_n         = 10 (constant in this experiment; kept for clarity)

That's 2 × 2 × 2 × 2 × 2 = 32 cells in principle. The full snippet variants
are densely populated; passage variants are sparse (smoke MAX_KW=20 only).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


VARIANT_AXES = {
    "biased":           ("biased",  "snippet"),
    "neutral":          ("neutral", "snippet"),
    "biased_passage":   ("biased",  "passage"),
    "neutral_passage":  ("neutral", "passage"),
}


def main() -> int:
    root = Path(os.getenv("GEODML_DATA_ROOT", REPO_ROOT / "geodml_data")).resolve()
    main_dir = root / "data" / "main"

    parts = []
    for variant, (prompt, passage_mode) in VARIANT_AXES.items():
        p = main_dir / f"full_experiment_data_{variant}.parquet"
        if not p.exists():
            print(f"  skip {variant}: {p} missing")
            continue
        df = pd.read_parquet(p)
        # Promote run-id-derived axes to top-level columns
        df = df.rename(columns={
            "search_engine": "axis_engine",
            "llm_model":     "axis_model",
            "pool":           "axis_pool",
            "top_n":         "axis_top_n",
        })
        df["axis_prompt"]       = prompt
        df["axis_passage_mode"] = passage_mode
        df["axis_variant"]      = variant  # backward compat
        # Drop redundant pre-existing prompt_variant if present
        if "prompt_variant" in df.columns:
            df = df.drop(columns=["prompt_variant"])
        # Dedup any column-name collisions from concat
        df = df.loc[:, ~df.columns.duplicated()]
        parts.append(df)
        print(f"  loaded {variant}: {len(df):,} rows")

    if not parts:
        print("nothing to merge", file=sys.stderr)
        return 2

    unified = pd.concat(parts, ignore_index=True)

    # Column ordering: axes → outcomes → treatments → confounders → meta
    axis_cols = [c for c in unified.columns if c.startswith("axis_")]
    outcome_cols = [c for c in ("pre_rank", "post_rank", "rank_delta") if c in unified.columns]
    treatment_cols = sorted([c for c in unified.columns if c.startswith("treat_")])
    confounder_cols = sorted([c for c in unified.columns if c.startswith("conf_") or c.startswith("dfs_")])
    id_cols = [c for c in ("keyword", "url", "domain", "run_id") if c in unified.columns]
    meta_cols = [c for c in unified.columns
                 if c not in axis_cols + outcome_cols + treatment_cols
                 + confounder_cols + id_cols and c not in ("axis_variant",)]

    # axis_variant is already in axis_cols (it starts with axis_); avoid dupe
    final_cols = list(dict.fromkeys(
        id_cols + axis_cols + outcome_cols + treatment_cols + confounder_cols + meta_cols
    ))
    unified = unified[[c for c in final_cols if c in unified.columns]]
    # Final dedup safety
    unified = unified.loc[:, ~unified.columns.duplicated()]

    out = main_dir / "full_experiment_unified.parquet"
    unified.to_parquet(out, index=False)

    print(f"\nwrote {len(unified):,} rows × {unified.shape[1]} cols → {out}")
    print()

    # ── Inventory ────────────────────────────────────────────────────────
    print("=" * 78)
    print("AXES OF VARIATION (8 columns)")
    print("=" * 78)
    for c in id_cols:
        print(f"  id          {c:25s} → {unified[c].nunique():,} unique values")
    for c in axis_cols + ["axis_variant"]:
        vals = sorted(unified[c].dropna().unique().tolist())
        if len(vals) <= 8:
            print(f"  axis        {c:25s} → {vals}")
        else:
            print(f"  axis        {c:25s} → {len(vals)} unique values")

    print()
    print("=" * 78)
    print(f"OUTCOMES ({len(outcome_cols)})")
    print("=" * 78)
    for c in outcome_cols:
        nn = unified[c].notna().sum()
        print(f"  {c:25s} non-null {nn:,}/{len(unified):,} ({nn/len(unified)*100:.1f}%)")

    print()
    print("=" * 78)
    print(f"TREATMENTS ({len(treatment_cols)})")
    print("=" * 78)
    for c in treatment_cols:
        nn = unified[c].notna().sum()
        is_binary = unified[c].dropna().nunique() <= 2
        kind = "binary " if is_binary else "cont.  "
        print(f"  {kind} {c:30s} non-null {nn:,}/{len(unified):,} ({nn/len(unified)*100:.0f}%)")

    print()
    print("=" * 78)
    print(f"CONFOUNDERS ({len(confounder_cols)})")
    print("=" * 78)
    for c in confounder_cols:
        nn = unified[c].notna().sum()
        kind = "DataForSEO" if c.startswith("dfs_") else \
               ("Moz/SEO   " if c in ("conf_domain_authority", "conf_backlinks", "conf_referring_domains") else \
                ("HTML      " if c in ("conf_word_count", "conf_readability", "conf_internal_links",
                                       "conf_outbound_links", "conf_images_alt") else \
                 ("SERP      " if c in ("conf_serp_position", "conf_title_kw_sim",
                                        "conf_snippet_kw_sim", "conf_title_len", "conf_snippet_len",
                                        "conf_title_has_kw", "conf_brand_recog", "conf_https",
                                        "conf_bm25") else "??        ")))
        print(f"  {kind} {c:32s} non-null {nn:,}/{len(unified):,} ({nn/len(unified)*100:.0f}%)")

    print()
    print("=" * 78)
    print("CELL COVERAGE (axis_prompt × axis_passage_mode × axis_engine × axis_model × axis_pool)")
    print("=" * 78)
    cells = unified.groupby(["axis_prompt", "axis_passage_mode", "axis_engine",
                             "axis_model", "axis_pool"]).size().rename("rows")
    keywords = unified.groupby(["axis_prompt", "axis_passage_mode", "axis_engine",
                                "axis_model", "axis_pool"])["keyword"].nunique().rename("keywords")
    rank_delta_nn = unified.groupby(["axis_prompt", "axis_passage_mode", "axis_engine",
                                     "axis_model", "axis_pool"])["rank_delta"].apply(
        lambda s: s.notna().sum()).rename("rank_delta_non_null")
    cell_summary = pd.concat([cells, keywords, rank_delta_nn], axis=1).reset_index()
    print(cell_summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
