#!/usr/bin/env python3
"""Reuse the upstream paperSizeExperiment's per-run phase3/features_new.parquet
files as the Stage B output, instead of re-extracting features from cached
HTML at ~25% coverage.

Each upstream cell (engine, model, pool) produced its own features_new.parquet
with all treatments + confounders populated. We union them per (engine, pool)
to maximize URL coverage, then write to
``data/features/features_<engine>_top<pool>.parquet`` where Stage C looks
for them.

Also extracts keyword-level DataForSEO confounders into
``data/features/dfs_keyword_confounders.parquet`` for the
--external-features-parquet flag of build_main_table.py.

Usage:
    python scripts/build_features_from_legacy.py
    GEODML_DATA_ROOT=... python scripts/build_features_from_legacy.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    root = Path(os.getenv("GEODML_DATA_ROOT", REPO_ROOT / "geodml_data")).resolve()
    runs = root / "data" / "runs"
    feat_dir = root / "data" / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    print(f"GEODML_DATA_ROOT = {root}")

    # Map new-pipeline engine names → legacy upstream engine names
    engine_map = {"searxng": ["searxng"], "ddg": ["ddg", "duckduckgo"]}

    # Stage B equivalent: union legacy phase3/features_new.parquet across both
    # models for each (engine, pool) — they share URLs because the cache was
    # populated from the same SERP regardless of which model was reranking.
    for engine_new, engine_legacy_names in engine_map.items():
        for pool in (20, 50):
            parts = []
            for model in ("Llama-3.3-70B-Instruct", "Qwen2.5-72B-Instruct"):
                for legacy in engine_legacy_names:
                    p = runs / f"{legacy}_{model}_serp{pool}_top10" / "phase3" / "features_new.parquet"
                    if p.exists():
                        df = pd.read_parquet(p)
                        df["_source_run"] = p.parent.parent.name
                        parts.append(df)
                        break  # first matching legacy name wins per model
            if not parts:
                print(f"  skip {engine_new}/pool={pool}: no upstream phase3 found")
                continue
            df = pd.concat(parts, ignore_index=True)
            # Dedup on (keyword, url) — keep first occurrence (Llama precedes Qwen)
            df = df.drop_duplicates(subset=["keyword", "url"], keep="first")
            # Drop the tracking column before writing
            df = df.drop(columns=["_source_run"], errors="ignore")
            out = feat_dir / f"features_{engine_new}_top{pool}.parquet"
            df.to_parquet(out, index=False)
            non_null = {c: df[c].notna().sum() for c in df.columns if c.startswith("treat_") or c.startswith("conf_")}
            n_treat = sum(1 for k, v in non_null.items() if k.startswith("treat_") and v > 0)
            n_conf = sum(1 for k, v in non_null.items() if k.startswith("conf_") and v > 0)
            print(f"  ✓ {out.name}: {len(df):,} rows × {df.shape[1]} cols  "
                  f"(treatments populated={n_treat}, confounders populated={n_conf})")

    # DataForSEO keyword-level confounders → external parquet for merge.py
    print("\nBuilding DataForSEO keyword confounders → data/features/dfs_keyword_confounders.parquet")
    ko_path = root / "data" / "dataforseo" / "keyword_overview.parquet"
    si_path = root / "data" / "dataforseo" / "search_intent.parquet"
    bkd_path = root / "data" / "dataforseo" / "bulk_keyword_difficulty.parquet"
    sv_path = root / "data" / "dataforseo" / "google_ads_search_volume.parquet"

    if not ko_path.exists():
        print(f"  no {ko_path}; skipping dfs_* confounders")
        return 0

    ko = pd.read_parquet(ko_path)
    # Pull the canonical fields from keyword_overview (uses ko.* aliases)
    out = pd.DataFrame({"keyword": ko["keyword"]})
    if "ko.search_volume" in ko.columns:
        out["dfs_search_volume"] = ko["ko.search_volume"]
    if "ko.cpc" in ko.columns:
        out["dfs_cpc"] = ko["ko.cpc"]
    if "ko.competition" in ko.columns:
        out["dfs_competition"] = ko["ko.competition"]
    if "ko.keyword_difficulty" in ko.columns:
        out["dfs_keyword_difficulty"] = ko["ko.keyword_difficulty"]

    # Search intent — one-hot the four categories from search_intent.parquet
    if si_path.exists():
        si = pd.read_parquet(si_path)
        # The intent file typically has keyword + main_intent or intent columns
        intent_col = next(
            (c for c in si.columns if c.lower().endswith("main_intent") and "prob" not in c.lower()),
            None,
        )
        if intent_col is None:
            intent_col = next((c for c in si.columns if c == "intent"), None)
        if intent_col is not None:
            si_lite = si[["keyword", intent_col]].drop_duplicates(subset=["keyword"])
            for label in ("commercial", "informational", "navigational", "transactional"):
                col = f"dfs_intent_{label}"
                si_lite[col] = (si_lite[intent_col].astype(str).str.lower() == label).astype(int)
            out = out.merge(
                si_lite[["keyword"] + [f"dfs_intent_{l}" for l in ("commercial", "informational", "navigational", "transactional")]],
                on="keyword", how="left",
            )

    # Backfill anything missing from bulk_keyword_difficulty if available
    if bkd_path.exists() and "dfs_keyword_difficulty" not in out.columns:
        bkd = pd.read_parquet(bkd_path)
        if "keyword" in bkd.columns and "keyword_difficulty" in bkd.columns:
            out = out.merge(
                bkd[["keyword", "keyword_difficulty"]].rename(columns={"keyword_difficulty": "dfs_keyword_difficulty"}),
                on="keyword", how="left",
            )

    out = out.drop_duplicates(subset=["keyword"])
    out_path = feat_dir / "dfs_keyword_confounders.parquet"
    out.to_parquet(out_path, index=False)
    n_kw = len(out)
    print(f"  ✓ {out_path.name}: {n_kw:,} keywords × {out.shape[1]} cols")
    print(f"    columns: {[c for c in out.columns if c != 'keyword']}")
    print(f"    coverage:")
    for c in [c for c in out.columns if c.startswith("dfs_")]:
        print(f"      {c}: {out[c].notna().sum():,}/{n_kw:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
