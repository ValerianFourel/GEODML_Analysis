"""Analyze order-probe outputs: compute pairwise overlap of LLM top-K under
different input orderings (original vs shuffled).

Reads:
    geodml_data/data/runs/{run_id}/phase2/keywords.jsonl   (original)
    geodml_data/data/order_probe/{run_id}_seed{S}.jsonl    (one per shuffle)

Where ``run_id = "{engine}_{ModelTag}_serp{N}_top{K}_{variant}"``.

Writes:
    geodml_data/data/order_probe/order_probe_summary.parquet

with columns:
    variant, model, engine, pool, keyword, K, ordering_pair,
    jaccard, overlap_at_k, n_a, n_b

Where ``ordering_pair`` ∈
{``orig_vs_seed42``, ``orig_vs_seed123``, ``seed42_vs_seed123``, ...}.

For each keyword present in BOTH files of a pair, we compute (at multiple K
values, default 3/5/10):

  set_a = ranked_domains[:K] from file A
  set_b = ranked_domains[:K] from file B
  jaccard      = |a ∩ b| / |a ∪ b|
  overlap_at_k = |a ∩ b| / K
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from interpretability.pipeline import config as C
from interpretability.utils import data_root


def _read_jsonl_top_k(path: Path) -> dict[str, list[str]]:
    """Return {keyword: ranked_domains} from a rerank/order-probe JSONL."""
    out: dict[str, list[str]] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            kw = rec.get("keyword")
            if not kw:
                continue
            out[kw] = list(rec.get("ranked_domains") or [])
    return out


def _pair_overlap(a: list[str], b: list[str], k: int) -> tuple[float, float, int, int]:
    """Return (jaccard, overlap_at_k, |a|, |b|) at top-K."""
    sa = set(a[:k])
    sb = set(b[:k])
    inter = len(sa & sb)
    union = len(sa | sb)
    jacc = inter / union if union else 0.0
    oak = inter / k if k else 0.0
    return jacc, oak, len(sa), len(sb)


def _enumerate_cells(variants: list[str]) -> list[dict]:
    cells: list[dict] = []
    for variant in variants:
        for model_id in C.LLM_MODELS:
            model_short = C.short_model_name(model_id)
            for engine in C.ENGINES:
                for serp_n, top_n in C.POOL_SIZES:
                    cells.append({
                        "variant": variant,
                        "model": model_short,
                        "engine": engine,
                        "pool": serp_n,
                        "top_n": top_n,
                        "run_id": C.run_label_with_variant(
                            engine, model_id, serp_n, top_n, variant,
                        ),
                    })
    return cells


def compute_summary(root: Path, variants: list[str], seeds: list[int],
                    k_values: list[int]) -> pd.DataFrame:
    rows: list[dict] = []
    for cell in _enumerate_cells(variants):
        run_id = cell["run_id"]
        orig_path = root / "data" / "runs" / run_id / "phase2" / "keywords.jsonl"
        seed_paths = {
            s: root / "data" / "order_probe" / f"{run_id}_seed{s}.jsonl"
            for s in seeds
        }

        sources: dict[str, dict[str, list[str]]] = {
            "orig": _read_jsonl_top_k(orig_path),
        }
        for s in seeds:
            sources[f"seed{s}"] = _read_jsonl_top_k(seed_paths[s])

        # Pairwise comparisons: (orig, seedX), (seedX, seedY).
        labels = list(sources.keys())
        for a_label, b_label in itertools.combinations(labels, 2):
            a_map = sources[a_label]
            b_map = sources[b_label]
            shared = set(a_map) & set(b_map)
            if not shared:
                continue
            pair = f"{a_label}_vs_{b_label}"
            for kw in shared:
                a = a_map[kw]
                b = b_map[kw]
                for k in k_values:
                    jacc, oak, na, nb = _pair_overlap(a, b, k)
                    rows.append({
                        "variant": cell["variant"],
                        "model": cell["model"],
                        "engine": cell["engine"],
                        "pool": cell["pool"],
                        "keyword": kw,
                        "K": k,
                        "ordering_pair": pair,
                        "jaccard": jacc,
                        "overlap_at_k": oak,
                        "n_a": na,
                        "n_b": nb,
                    })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=["biased", "neutral"],
                    choices=("biased", "neutral"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    ap.add_argument("--K", nargs="+", type=int, default=[3, 5, 10])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output", default=None,
                    help="Override output parquet path (default: data/order_probe/order_probe_summary.parquet).")
    args = ap.parse_args()

    root = data_root(args.data_root)
    df = compute_summary(root, args.variants, args.seeds, args.K)

    out = Path(args.output) if args.output else (
        root / "data" / "order_probe" / "order_probe_summary.parquet"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print(f"[order_probe_analyze] no overlapping keywords found across pairs. "
              f"Variants={args.variants} seeds={args.seeds}. "
              f"Did the order-probe runs finish?")
        # Still write an empty parquet so downstream code can detect "ran but empty".
        df.to_parquet(out, index=False)
        return 1

    df.to_parquet(out, index=False)
    print(f"[order_probe_analyze] wrote {len(df):,} rows -> {out}")

    # Headline summary: mean overlap@10 by (variant, ordering_pair).
    head = df[df.K == 10].groupby(["variant", "ordering_pair"]).agg(
        mean_jacc=("jaccard", "mean"),
        mean_oak=("overlap_at_k", "mean"),
        n=("keyword", "count"),
    ).round(3)
    print("\n[order_probe_analyze] headline (K=10):")
    print(head.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
