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


def biased_minus_neutral(cell: pd.DataFrame) -> pd.DataFrame:
    pivot = cell.pivot_table(
        index=["model", "engine", "pool"],
        columns="variant",
        values=["mean_jacc", "mean_oak"],
    )
    if "biased" not in pivot["mean_oak"].columns or "neutral" not in pivot["mean_oak"].columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        "biased_jacc":  pivot["mean_jacc"]["biased"],
        "neutral_jacc": pivot["mean_jacc"]["neutral"],
        "delta_jacc":   pivot["mean_jacc"]["biased"] - pivot["mean_jacc"]["neutral"],
        "biased_oak":   pivot["mean_oak"]["biased"],
        "neutral_oak":  pivot["mean_oak"]["neutral"],
        "delta_oak":    pivot["mean_oak"]["biased"] - pivot["mean_oak"]["neutral"],
    }).round(3).reset_index().sort_values("delta_oak").reset_index(drop=True)
    return out


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
    trend = k_trend(df)
    worst = worst_keywords(df, args.k, args.worst_n)

    cell_csv  = outdir / "order_probe_by_cell.csv"
    delta_csv = outdir / "order_probe_biased_minus_neutral.csv"
    trend_csv = outdir / "order_probe_k_trend.csv"
    worst_csv = outdir / "order_probe_worst_keywords.csv"

    cell.to_csv(cell_csv, index=False)
    if not delta.empty:
        delta.to_csv(delta_csv, index=False)
    trend.to_csv(trend_csv, index=False)
    worst.to_csv(worst_csv, index=False)

    print(f"[order_probe_report] read {len(df):,} rows from {summary_path}")
    print(f"[order_probe_report] wrote {cell_csv}")
    if not delta.empty:
        print(f"[order_probe_report] wrote {delta_csv}")
    print(f"[order_probe_report] wrote {trend_csv}")
    print(f"[order_probe_report] wrote {worst_csv}")

    print(f"\n=== per-cell at K={args.k} (mean over keywords × ordering pairs) ===")
    print(cell.to_string(index=False))

    if not delta.empty:
        print("\n=== biased − neutral gap (K={}, sorted by Δ overlap@K asc) ===".format(args.k))
        print(delta.to_string(index=False))

    print("\n=== K-trend (averaged over all cells of a variant) ===")
    print(trend.to_string(index=False))

    print(f"\n=== {args.worst_n} keywords with the lowest mean overlap@{args.k} ===")
    print(worst.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
