"""Stage C - merge per-run rerank outputs + per-(engine, pool) features into
a single DML-ready parquet, one per prompt variant.

Inputs:
    geodml_data/data/runs/{run_id}/phase2/keywords.jsonl   (from rerank.py)
    geodml_data/data/features/features_{engine}_top{pool}.parquet  (from features.py)

Outputs:
    geodml_data/data/main/full_experiment_data_{variant}.parquet

Optionally left-joins external feature columns (Moz, OpenPageRank,
DataForSEO) from a precomputed parquet via ``--external-features-parquet``.
Those columns will simply be NaN if not provided; DML drops NaN-only columns.

Ported from:
    pipeline/clean_data.py:main
    paperSizeExperiment/run_experiment.py:merge_all_datasets
    (the pure-pandas merge logic; HTTP / Moz fetching dropped).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from interpretability.pipeline import config as C
from interpretability.utils import data_root


# ── Run-id parsing ───────────────────────────────────────────────────────────

_RUN_RE = re.compile(
    r"^(?P<engine>[^_]+)_(?P<model>.+)_serp(?P<pool>\d+)_top(?P<topn>\d+)"
    r"_(?P<variant>biased_passage|neutral_passage|biased_rag|neutral_rag|biased|neutral)$"
)


def parse_run_id(run_id: str) -> dict:
    m = _RUN_RE.match(run_id)
    if not m:
        raise ValueError(f"unparseable run_id: {run_id!r}")
    g = m.groupdict()
    return {
        "engine":  g["engine"],
        "model":   g["model"],
        "pool":    int(g["pool"]),
        "top_n":   int(g["topn"]),
        "variant": g["variant"],
    }


def list_variant_runs(variant: str, root: Path) -> list[str]:
    runs_dir = root / "data" / "runs"
    if not runs_dir.exists():
        return []
    out: list[str] = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        if not d.name.endswith(f"_{variant}"):
            continue
        if not (d / "phase2" / "keywords.jsonl").exists():
            continue
        out.append(d.name)
    return out


# ── Per-run merge ────────────────────────────────────────────────────────────

def merge_one_run(run_id: str, root: Path) -> pd.DataFrame:
    """Build a per-run dataframe of (keyword, url, domain, treat_*, conf_*, rank_*)."""
    meta = parse_run_id(run_id)

    jsonl_path = root / "data" / "runs" / run_id / "phase2" / "keywords.jsonl"
    feat_path  = root / "data" / "features" / f"features_{meta['engine']}_top{meta['pool']}.parquet"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"missing rerank output: {jsonl_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"missing features parquet: {feat_path}")

    feat = pd.read_parquet(feat_path)

    # Build (keyword, url) -> feature row index (faster than apply)
    feat_idx = {(r["keyword"], r["url"]): i for i, r in feat.iterrows()}
    # Also index by (keyword, domain) for fallback when URL doesn't match
    feat_dom_idx: dict[tuple[str, str], int] = {}
    for i, r in feat.iterrows():
        key = (r["keyword"], r["domain"])
        # First-seen wins
        feat_dom_idx.setdefault(key, i)

    rows: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            kw = rec.get("keyword") or rec.get("query") or ""
            ranked = {r["domain"]: r["url"] for r in rec.get("ranked_results", [])}
            llm_params = rec.get("llm_parameters") or {}
            llm_backend = llm_params.get("backend")
            llm_precision = llm_params.get("precision")
            for rc in rec.get("rank_changes", []):
                domain = rc["domain"]
                url = ranked.get(domain, "")

                # Look up features by (kw, url) first, else (kw, domain).
                feat_row = None
                if url and (kw, url) in feat_idx:
                    feat_row = feat.iloc[feat_idx[(kw, url)]]
                elif (kw, domain) in feat_dom_idx:
                    feat_row = feat.iloc[feat_dom_idx[(kw, domain)]]

                row = {
                    "run_id":         run_id,
                    "search_engine":  meta["engine"],
                    "llm_model":      meta["model"],
                    "llm_backend":    llm_backend,
                    "llm_precision":  llm_precision,
                    "pool":           meta["pool"],
                    "top_n":          meta["top_n"],
                    "prompt_variant": meta["variant"],
                    "keyword":        kw,
                    "domain":         domain,
                    "url":            url,
                    "pre_rank":       rc.get("pre_rank"),
                    "post_rank":      rc.get("post_rank"),
                    "rank_delta":     rc.get("rank_delta"),
                    "used_fallback":  rec.get("used_fallback", False),
                }
                # Attach all feature columns (treat_*, conf_*, etc.)
                if feat_row is not None:
                    for col, val in feat_row.items():
                        if col in ("keyword", "url", "domain"):
                            continue  # already in row
                        row[col] = val
                rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ── Optional external join ───────────────────────────────────────────────────

def left_join_external(df: pd.DataFrame, ext_path: Path) -> pd.DataFrame:
    if not ext_path.exists():
        print(f"[merge] external features parquet not found: {ext_path} (skipping)",
              flush=True)
        return df
    ext = pd.read_parquet(ext_path)
    join_keys = [k for k in ("keyword", "url", "domain") if k in ext.columns]
    if not join_keys:
        print(f"[merge] external parquet has no join keys; skipping", flush=True)
        return df
    print(f"[merge] left-joining {len(ext.columns) - len(join_keys)} external columns "
          f"on {join_keys} from {ext_path}", flush=True)
    return df.merge(ext, on=join_keys, how="left", suffixes=("", "_ext"))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage C: merge rerank + features into full_experiment_data_{variant}.parquet",
    )
    ap.add_argument("--variant", required=True,
                    choices=(
                        "biased", "neutral",
                        "biased_passage", "neutral_passage",
                        "biased_rag", "neutral_rag",
                    ))
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Subset of run_ids to include. Default: all "
                         "{engine}_{model}_serp{pool}_top{top_n}_{variant} "
                         "matching variant.")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--external-features-parquet", default=None,
                    help="Optional pre-computed parquet (Moz/PageRank/DfS columns) "
                         "to left-join on (keyword, url) or (keyword, domain).")
    ap.add_argument("--output", default=None,
                    help="Override output path. Default: data/main/full_experiment_data_{variant}.parquet")
    args = ap.parse_args()

    root = data_root(args.data_root)

    runs = args.runs or list_variant_runs(args.variant, root)
    if not runs:
        print(f"[merge] FATAL: no runs found for variant={args.variant}",
              file=sys.stderr)
        return 2
    print(f"[merge] merging {len(runs)} runs for variant={args.variant}:", flush=True)
    for r in runs:
        print(f"        {r}", flush=True)

    parts: list[pd.DataFrame] = []
    for run_id in runs:
        try:
            d = merge_one_run(run_id, root)
            print(f"[merge] {run_id}: {len(d):,} rows", flush=True)
            if len(d):
                parts.append(d)
        except FileNotFoundError as e:
            print(f"[merge] SKIP {run_id}: {e}", flush=True)
            continue

    if not parts:
        print("[merge] FATAL: nothing to merge", file=sys.stderr)
        return 2

    df = pd.concat(parts, ignore_index=True)

    if args.external_features_parquet:
        df = left_join_external(df, Path(args.external_features_parquet))

    out = Path(args.output) if args.output else C.main_table_path(args.variant, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"[merge] wrote {len(df):,} rows ({df.shape[1]} cols) -> {out}",
          flush=True)
    print(f"[merge] keywords={df['keyword'].nunique():,} "
          f"runs={df['run_id'].nunique()} "
          f"models={df['llm_model'].nunique()}", flush=True)
    if "llm_precision" in df.columns:
        precision_breakdown = (
            df["llm_precision"].fillna("(missing)").value_counts().to_dict()
        )
        print(f"[merge] llm_precision breakdown: {precision_breakdown}", flush=True)
    if "rank_delta" in df.columns:
        n_rd = df["rank_delta"].notna().sum()
        print(f"[merge] rank_delta non-null: {n_rd:,}/{len(df):,} ({n_rd/len(df)*100:.1f}%)",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
