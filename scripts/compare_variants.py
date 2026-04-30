"""Compare two rerank variants on identical (cell × keyword) inputs.

Computes per-keyword agreement metrics — Kendall's τ on the rank-vector
restricted to the intersection, RBO@K with persistence p (Webber et al. 2010),
and Jaccard@K — between any two ``run_id``s with overlapping ``keywords.jsonl``
records.

Usage:

    python scripts/compare_variants.py \\
        --run-a searxng_Llama-3.3-70B-Instruct_serp50_top10_biased \\
        --run-b searxng_Llama-3.3-70B-Instruct_serp50_top10_biased_passage

    # The four headline diagnostics for the bracketing experiment
    python scripts/compare_variants.py --pair-grid

Outputs:
    geodml_data/data/comparisons/{run_a}__vs__{run_b}.parquet
    Per-keyword: kendall_tau, kendall_p, rbo_at_k, jaccard_at_k, n_a, n_b, n_int.
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from interpretability.pipeline import config as C
from interpretability.utils import data_root


def _load_ranked_domains(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[str, list[str]] = {}
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


def rbo_at_k(a: list[str], b: list[str], k: int, p: float = 0.9) -> float:
    """Rank-Biased Overlap @ K (Webber, Moffat & Zobel, 2010), persistence p.

    Bounded in [0, 1]; 1 = identical prefixes, 0 = disjoint. Uses the
    extrapolated form for finite top-K (eq. 32 in the original paper),
    treating positions beyond the prefix as missing.
    """
    if k <= 0:
        return 0.0
    a_pref = a[:k]
    b_pref = b[:k]
    set_a: set[str] = set()
    set_b: set[str] = set()
    overlap = 0
    weighted_sum = 0.0
    for d in range(1, k + 1):
        if d - 1 < len(a_pref):
            x = a_pref[d - 1]
            if x in set_b:
                overlap += 1
            set_a.add(x)
        if d - 1 < len(b_pref):
            y = b_pref[d - 1]
            if y in set_a:
                overlap += 1
            set_b.add(y)
        # When both lists yield the same item this depth, the symmetric
        # increment above counts it twice. Adjust:
        if (d - 1 < len(a_pref) and d - 1 < len(b_pref)
                and a_pref[d - 1] == b_pref[d - 1]):
            overlap -= 1
        agreement_d = overlap / d
        weighted_sum += (p ** (d - 1)) * agreement_d
    return (1.0 - p) * weighted_sum + (p ** k) * (overlap / k)


def kendall_on_intersection(a: list[str], b: list[str]) -> tuple[float, float, int]:
    """Kendall's τ-b on the rank vector restricted to common items.

    Returns (tau, p_value, n_intersection). When |intersection| < 2, returns
    (nan, nan, |intersection|).
    """
    inter = [d for d in a if d in set(b)]
    n = len(inter)
    if n < 2:
        return float("nan"), float("nan"), n
    rank_a = [a.index(d) for d in inter]
    rank_b = [b.index(d) for d in inter]
    tau, p = kendalltau(rank_a, rank_b)
    return float(tau), float(p), n


def compare(
    run_a: str, run_b: str, root: Path, k: int = 10, p: float = 0.9,
) -> pd.DataFrame:
    a_path = root / "data" / "runs" / run_a / "phase2" / "keywords.jsonl"
    b_path = root / "data" / "runs" / run_b / "phase2" / "keywords.jsonl"
    a_map = _load_ranked_domains(a_path)
    b_map = _load_ranked_domains(b_path)
    shared = sorted(set(a_map) & set(b_map))
    rows: list[dict] = []
    for kw in shared:
        a = a_map[kw]
        b = b_map[kw]
        if not a or not b:
            continue
        tau, p_val, n_int = kendall_on_intersection(a, b)
        rbo = rbo_at_k(a, b, k=k, p=p)
        jacc_set_a = set(a[:k])
        jacc_set_b = set(b[:k])
        union = len(jacc_set_a | jacc_set_b)
        jacc = (len(jacc_set_a & jacc_set_b) / union) if union else 0.0
        rows.append({
            "keyword":      kw,
            "n_a":          len(a),
            "n_b":          len(b),
            "n_int":        n_int,
            "kendall_tau":  tau,
            "kendall_p":    p_val,
            "rbo_at_k":     rbo,
            "jaccard_at_k": jacc,
        })
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                  rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Mean + 95% bootstrap CI. Drops NaNs."""
    rng = rng or np.random.default_rng(42)
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return (float("nan"), float("nan"), float("nan"))
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def _summarize(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        print(f"  [{label}]  (no overlapping keywords)")
        return
    cols = ("kendall_tau", "rbo_at_k", "jaccard_at_k")
    print(f"  [{label}]  n_keywords={len(df):,}")
    for c in cols:
        m, lo, hi = _bootstrap_ci(df[c].to_numpy(dtype=float))
        print(f"    {c:<14s}  mean={m:+.3f}  95% CI=[{lo:+.3f}, {hi:+.3f}]")


def _enumerate_pair_grid() -> list[tuple[str, str]]:
    """The four diagnostics: per cell, four pair comparisons."""
    pairs: list[tuple[str, str]] = []
    for model_id, engine, (serp_n, top_n) in product(
        C.LLM_MODELS, C.ENGINES, C.POOL_SIZES,
    ):
        rid = lambda v: C.run_label_with_variant(engine, model_id, serp_n, top_n, v)
        # 1) passage effect inside biased
        pairs.append((rid("biased"),         rid("biased_passage")))
        # 2) passage effect inside neutral
        pairs.append((rid("neutral"),        rid("neutral_passage")))
        # 3) prompt-instruction effect, snippet-only (sanity, should reproduce)
        pairs.append((rid("biased"),         rid("neutral")))
        # 4) prompt-instruction effect, passage-augmented
        pairs.append((rid("biased_passage"), rid("neutral_passage")))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-a", help="First run_id (e.g. searxng_..._biased).")
    ap.add_argument("--run-b", help="Second run_id (compared against --run-a).")
    ap.add_argument("--pair-grid", action="store_true",
                    help="Compute the four headline pair comparisons across all "
                         "cells (8 cells × 4 pairs = 32 comparisons). Overrides "
                         "--run-a/--run-b.")
    ap.add_argument("--k", type=int, default=10, help="Top-K (default 10).")
    ap.add_argument("--p", type=float, default=0.9, help="RBO persistence (default 0.9).")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--outdir", default=None,
                    help="Override comparisons output dir "
                         "(default: data/comparisons/).")
    args = ap.parse_args()

    root = data_root(args.data_root)
    outdir = Path(args.outdir) if args.outdir else root / "data" / "comparisons"
    outdir.mkdir(parents=True, exist_ok=True)

    if args.pair_grid:
        pairs = _enumerate_pair_grid()
    else:
        if not (args.run_a and args.run_b):
            ap.error("must pass either --pair-grid OR both --run-a and --run-b")
        pairs = [(args.run_a, args.run_b)]

    print(f"[compare_variants] K={args.k} p={args.p} pairs={len(pairs)}")
    for run_a, run_b in pairs:
        try:
            df = compare(run_a, run_b, root, k=args.k, p=args.p)
        except FileNotFoundError as e:
            print(f"  SKIP {run_a} vs {run_b}  (missing: {e})")
            continue
        out_path = outdir / f"{run_a}__vs__{run_b}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"\n{run_a}\n  vs {run_b}")
        _summarize(df, f"{run_a} vs {run_b}")
        print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
