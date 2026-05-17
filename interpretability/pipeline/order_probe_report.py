"""Slice ``order_probe_summary.parquet`` into operator-friendly tables.

Reads:
    geodml_data/data/order_probe/order_probe_summary.parquet

Writes (to ``interpretability/output/`` by default):
    order_probe_by_cell.csv          per (variant, model, engine, pool) at K=10
    order_probe_biased_minus_neutral.csv   per (model, engine, pool): biased − neutral gap
    order_probe_k_trend.csv          mean overlap by (variant, K)
    order_probe_worst_keywords.csv   N keywords with the lowest mean overlap@10

Also prints the same tables to stdout so the job log is self-contained.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from interpretability.utils import data_root

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = REPO_ROOT / "interpretability" / "output"


def _load(parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    if df.empty:
        raise SystemExit(f"[order_probe_report] {parquet} is empty — run order_probe_analyze first.")
    return df


def by_cell(df: pd.DataFrame, k: int) -> pd.DataFrame:
    sub = df[df.K == k]
    out = (
        sub.groupby(["variant", "model", "engine", "pool"], as_index=False)
           .agg(
               mean_jacc=("jaccard", "mean"),
               mean_oak=("overlap_at_k", "mean"),
               p10_oak=("overlap_at_k", lambda s: s.quantile(0.10)),
               n_obs=("keyword", "count"),
               n_kw=("keyword", "nunique"),
           )
           .round(3)
           .sort_values(["variant", "model", "engine", "pool"])
           .reset_index(drop=True)
    )
    return out


def _pair_delta(cell: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Return one row per (model, engine, pool) with Δ = a − b for jacc and oak."""
    pivot = cell.pivot_table(
        index=["model", "engine", "pool"],
        columns="variant",
        values=["mean_jacc", "mean_oak"],
    )
    if a not in pivot["mean_oak"].columns or b not in pivot["mean_oak"].columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        f"{a}_jacc":  pivot["mean_jacc"][a],
        f"{b}_jacc":  pivot["mean_jacc"][b],
        "delta_jacc": pivot["mean_jacc"][a] - pivot["mean_jacc"][b],
        f"{a}_oak":   pivot["mean_oak"][a],
        f"{b}_oak":   pivot["mean_oak"][b],
        "delta_oak":  pivot["mean_oak"][a] - pivot["mean_oak"][b],
    }).round(3).reset_index().sort_values("delta_oak").reset_index(drop=True)
    return out


def biased_minus_neutral(cell: pd.DataFrame) -> pd.DataFrame:
    """biased − neutral, snippet-only (the original prompt-instruction effect)."""
    return _pair_delta(cell, "biased", "neutral")


def augmented_minus_snippet(cell: pd.DataFrame, prompt: str, suffix: str) -> pd.DataFrame:
    """{prompt}_{suffix} − {prompt}: per-prompt augmentation effect.

    ``suffix`` is "passage" (leading body) or "rag" (retrieved chunks).
    A negative ``delta_oak`` means the augmentation destabilizes the rerank;
    positive means it makes ranking more reproducible.
    """
    return _pair_delta(cell, f"{prompt}_{suffix}", prompt)


def passage_minus_snippet(cell: pd.DataFrame, prompt: str) -> pd.DataFrame:
    """Back-compat alias: {prompt}_passage − {prompt}."""
    return augmented_minus_snippet(cell, prompt, "passage")


def rag_minus_snippet(cell: pd.DataFrame, prompt: str) -> pd.DataFrame:
    """{prompt}_rag − {prompt}: per-prompt retrieval-augmentation effect."""
    return augmented_minus_snippet(cell, prompt, "rag")


def biased_minus_neutral_passage(cell: pd.DataFrame) -> pd.DataFrame:
    """biased_passage − neutral_passage (does the prompt-instruction gap survive
    passage augmentation?)."""
    return _pair_delta(cell, "biased_passage", "neutral_passage")


def biased_minus_neutral_rag(cell: pd.DataFrame) -> pd.DataFrame:
    """biased_rag − neutral_rag (does the prompt-instruction gap survive
    retrieval augmentation?)."""
    return _pair_delta(cell, "biased_rag", "neutral_rag")


def rag_minus_passage(cell: pd.DataFrame, prompt: str) -> pd.DataFrame:
    """{prompt}_rag − {prompt}_passage: retrieval-relevance effect, holding
    information amount roughly constant."""
    return _pair_delta(cell, f"{prompt}_rag", f"{prompt}_passage")


def k_trend(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["variant", "K"], as_index=False)
          .agg(mean_jacc=("jaccard", "mean"),
               mean_oak=("overlap_at_k", "mean"),
               n_obs=("keyword", "count"))
          .round(3)
          .sort_values(["variant", "K"])
          .reset_index(drop=True)
    )


def worst_keywords(df: pd.DataFrame, k: int, top_n: int) -> pd.DataFrame:
    sub = df[df.K == k]
    per_kw = (
        sub.groupby(["variant", "model", "engine", "pool", "keyword"], as_index=False)
           .agg(mean_oak=("overlap_at_k", "mean"),
                mean_jacc=("jaccard", "mean"),
                n_pairs=("ordering_pair", "nunique"))
    )
    per_kw = per_kw.sort_values("mean_oak").head(top_n).round(3).reset_index(drop=True)
    return per_kw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--summary", default=None,
                    help="Override path to order_probe_summary.parquet.")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR),
                    help=f"Where to write CSVs (default: {DEFAULT_OUTDIR}).")
    ap.add_argument("--k", type=int, default=10, help="Top-K to slice on (default 10).")
    ap.add_argument("--worst-n", type=int, default=30,
                    help="How many worst-overlap keywords to dump (default 30).")
    args = ap.parse_args()

    root = data_root(args.data_root)
    summary_path = Path(args.summary) if args.summary else (
        root / "data" / "order_probe" / "order_probe_summary.parquet"
    )
    df = _load(summary_path)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cell = by_cell(df, args.k)
    delta = biased_minus_neutral(cell)
    delta_biased_passage  = passage_minus_snippet(cell, "biased")
    delta_neutral_passage = passage_minus_snippet(cell, "neutral")
    delta_passage_prompt  = biased_minus_neutral_passage(cell)
    delta_biased_rag      = rag_minus_snippet(cell, "biased")
    delta_neutral_rag     = rag_minus_snippet(cell, "neutral")
    delta_rag_prompt      = biased_minus_neutral_rag(cell)
    delta_biased_rag_vs_passage  = rag_minus_passage(cell, "biased")
    delta_neutral_rag_vs_passage = rag_minus_passage(cell, "neutral")
    trend = k_trend(df)
    worst = worst_keywords(df, args.k, args.worst_n)

    cell_csv  = outdir / "order_probe_by_cell.csv"
    delta_csv = outdir / "order_probe_biased_minus_neutral.csv"
    delta_bp_csv = outdir / "order_probe_biased_passage_minus_biased.csv"
    delta_np_csv = outdir / "order_probe_neutral_passage_minus_neutral.csv"
    delta_pp_csv = outdir / "order_probe_biased_passage_minus_neutral_passage.csv"
    delta_br_csv = outdir / "order_probe_biased_rag_minus_biased.csv"
    delta_nr_csv = outdir / "order_probe_neutral_rag_minus_neutral.csv"
    delta_rr_csv = outdir / "order_probe_biased_rag_minus_neutral_rag.csv"
    delta_brp_csv = outdir / "order_probe_biased_rag_minus_biased_passage.csv"
    delta_nrp_csv = outdir / "order_probe_neutral_rag_minus_neutral_passage.csv"
    trend_csv = outdir / "order_probe_k_trend.csv"
    worst_csv = outdir / "order_probe_worst_keywords.csv"

    cell.to_csv(cell_csv, index=False)
    extras = (
        ("biased − neutral",                       delta,                 delta_csv),
        ("biased_passage − biased",                delta_biased_passage,  delta_bp_csv),
        ("neutral_passage − neutral",              delta_neutral_passage, delta_np_csv),
        ("biased_passage − neutral_passage",       delta_passage_prompt,  delta_pp_csv),
        ("biased_rag − biased",                    delta_biased_rag,      delta_br_csv),
        ("neutral_rag − neutral",                  delta_neutral_rag,     delta_nr_csv),
        ("biased_rag − neutral_rag",               delta_rag_prompt,      delta_rr_csv),
        ("biased_rag − biased_passage",            delta_biased_rag_vs_passage,  delta_brp_csv),
        ("neutral_rag − neutral_passage",          delta_neutral_rag_vs_passage, delta_nrp_csv),
    )
    for _label, frame, path in extras:
        if not frame.empty:
            frame.to_csv(path, index=False)
    trend.to_csv(trend_csv, index=False)
    worst.to_csv(worst_csv, index=False)

    print(f"[order_probe_report] read {len(df):,} rows from {summary_path}")
    print(f"[order_probe_report] wrote {cell_csv}")
    for _label, frame, path in extras:
        if not frame.empty:
            print(f"[order_probe_report] wrote {path}")
    print(f"[order_probe_report] wrote {trend_csv}")
    print(f"[order_probe_report] wrote {worst_csv}")

    print(f"\n=== per-cell at K={args.k} (mean over keywords × ordering pairs) ===")
    print(cell.to_string(index=False))

    for label, frame, _path in extras:
        if frame.empty:
            continue
        print(f"\n=== {label} gap (K={args.k}, sorted by Δ overlap@K asc) ===")
        print(frame.to_string(index=False))

    print("\n=== K-trend (averaged over all cells of a variant) ===")
    print(trend.to_string(index=False))

    print(f"\n=== {args.worst_n} keywords with the lowest mean overlap@{args.k} ===")
    print(worst.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
