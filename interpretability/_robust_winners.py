"""Robust-winners frame helper.

A *robust winner* is a (keyword, url) pair that the LLM placed in its top-10
under BOTH serp20 and serp50 pools within a single (search_engine, llm_model)
category. Conditioning the mech-interp experiments on this set excludes the
selection effect (does the model pick the page) and isolates the position
effect — the same conditioning that produced the headline DML coefficients
in `docs/robust-winners-analysis-2026-04-26.md` on the HF dataset.

Median per-category top-10 overlap across pools is only 2-3/10 (Jaccard
0.22-0.31), so this set is meaningfully smaller than the full table.

Public API:
    load_robust_winner_pairs() -> pd.DataFrame
        with columns [keyword, url, search_engine, llm_model]

The result is cached at `interpretability/output/_robust_pairs.parquet`. If
the cache is from today (mtime >= today 00:00), it is reused; otherwise it
is rederived from the local main parquet (or HF dataset as fallback).
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from pathlib import Path

import pandas as pd

from interpretability.utils import RUN_IDS, load_main_table

CACHE_PATH = Path(__file__).resolve().parent / "output" / "_robust_pairs.parquet"

# Category-level expected counts from docs/robust-winners-analysis-2026-04-26.md
# § 3 (rounded). The current sanity check is ±10%.
EXPECTED_COUNTS = {
    ("duckduckgo", "Llama-3.3-70B-Instruct"): 2187,
    ("duckduckgo", "Qwen2.5-72B-Instruct"): 2572,
    ("searxng", "Llama-3.3-70B-Instruct"): 2386,
    ("searxng", "Qwen2.5-72B-Instruct"): 3031,
}

_RUN_ID_RE = re.compile(
    r"^(?P<engine>duckduckgo|searxng|ddg)_(?P<model>.+?)_serp(?P<pool>\d+)_top\d+$"
)


def _split_run_id(run_id: str) -> tuple[str, str, int] | None:
    m = _RUN_ID_RE.match(str(run_id))
    if not m:
        return None
    engine = m.group("engine")
    if engine == "ddg":
        engine = "duckduckgo"
    return engine, m.group("model"), int(m.group("pool"))


def _ensure_category_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure search_engine / llm_model / serp_pool_size columns exist.

    If absent, parse them from `run_id` (every row in the main table carries
    a run_id of the form `{engine}_{model}_serp{pool}_top{N}`).

    Always normalises `llm_model` to the short form (no `org/` prefix) so the
    derived pairs match how callers look them up via `short_model_name()`.
    """
    needed = {"search_engine", "llm_model", "serp_pool_size"}
    if needed.issubset(df.columns):
        df = df.copy()
        df["llm_model"] = df["llm_model"].astype(str).map(short_model_name)
        return df
    if "run_id" not in df.columns:
        raise ValueError(
            "main table missing both run_id and category columns "
            "(search_engine, llm_model, serp_pool_size); cannot derive frames"
        )
    parsed = df["run_id"].map(_split_run_id)
    if parsed.isna().any():
        bad = df.loc[parsed.isna(), "run_id"].unique()[:5]
        raise ValueError(f"unparseable run_ids: {list(bad)}")
    df = df.copy()
    df["search_engine"] = parsed.map(lambda t: t[0])
    df["llm_model"] = parsed.map(lambda t: short_model_name(t[1]))
    df["serp_pool_size"] = parsed.map(lambda t: t[2]).astype(int)
    return df


def _derive_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_category_cols(df)
    keys = ["keyword", "url", "search_engine", "llm_model"]
    pools_seen = (
        df.groupby(keys)["serp_pool_size"]
        .agg(lambda s: frozenset(int(x) for x in s.unique()))
        .reset_index()
    )
    pools_seen["is_robust"] = pools_seen["serp_pool_size"].map(
        lambda S: 20 in S and 50 in S
    )
    return pools_seen.loc[pools_seen.is_robust, keys].reset_index(drop=True)


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    mtime = _dt.datetime.fromtimestamp(path.stat().st_mtime)
    today_start = _dt.datetime.combine(_dt.date.today(), _dt.time.min)
    return mtime >= today_start


def load_robust_winner_pairs(
    *, refresh: bool = False, verify: bool = True
) -> pd.DataFrame:
    """Returns a DataFrame with columns [keyword, url, search_engine, llm_model].

    Caches to `interpretability/output/_robust_pairs.parquet` for the day. Pass
    `refresh=True` to force rederivation. With `verify=True` (default), prints
    per-category counts and warns loudly if any category drifts >10% from the
    2026-04-26 reference.
    """
    if not refresh and _cache_is_fresh(CACHE_PATH):
        pairs = pd.read_parquet(CACHE_PATH)
        print(f"[robust] cache hit ({len(pairs)} pairs) -> {CACHE_PATH}")
    else:
        try:
            df = load_main_table()
            source = "local parquet"
        except Exception as e:
            print(f"[robust] local main parquet unavailable ({e}); trying HF dataset")
            from datasets import load_dataset

            df = load_dataset(
                "ValerianFourel/geodml-papersize", "main", split="train"
            ).to_pandas()
            source = "HF datasets"
        pairs = _derive_pairs(df)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so concurrent SLURM jobs racing the same fresh cache
        # never observe a partial parquet.
        tmp = CACHE_PATH.with_suffix(f".tmp.{os.getpid()}.parquet")
        pairs.to_parquet(tmp, index=False)
        os.replace(tmp, CACHE_PATH)
        print(f"[robust] derived {len(pairs)} pairs from {source} -> {CACHE_PATH}")

    if verify:
        counts = (
            pairs.groupby(["search_engine", "llm_model"]).size().to_dict()
        )
        print("[robust] per-category counts:")
        any_drift = False
        for cat in sorted(set(EXPECTED_COUNTS) | set(counts)):
            actual = counts.get(cat, 0)
            expected = EXPECTED_COUNTS.get(cat)
            if expected is None:
                print(f"  {cat[0]:11s}+{cat[1]:24s}  n={actual:5d}  (no expected)")
                continue
            tol = 0.10
            drift = abs(actual - expected) / expected if expected else 0.0
            flag = "OK" if drift <= tol else "DRIFT"
            print(
                f"  {cat[0]:11s}+{cat[1]:24s}  n={actual:5d}  "
                f"expected≈{expected:5d}  drift={drift:5.1%}  [{flag}]"
            )
            if drift > tol:
                any_drift = True
        if any_drift:
            print(
                "[robust] WARNING: at least one category drifted >10% from the "
                "2026-04-26 reference. The dataset may have changed; re-check "
                "the docs/robust-winners-analysis writeup before publishing."
            )

    return pairs


def filter_to_robust(
    df: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    search_engine: str | None = None,
    llm_model: str | None = None,
    require_kw_url: bool = True,
) -> pd.DataFrame:
    """Inner-join `df` against the robust pairs.

    By default joins on (keyword, url, search_engine, llm_model). Pass
    `search_engine` / `llm_model` to constrain the pair set first when `df`
    doesn't carry those columns (e.g. a SERP candidate dataframe). When
    `require_kw_url=False`, returns rows whose keyword alone has any robust
    winner — useful for keyword-level samplers.
    """
    p = pairs
    if search_engine is not None:
        p = p[p["search_engine"] == search_engine]
    if llm_model is not None:
        p = p[p["llm_model"] == llm_model]

    if not require_kw_url:
        kws = set(p["keyword"].unique())
        return df[df["keyword"].isin(kws)].copy()

    join_keys = ["keyword", "url"]
    if "search_engine" in df.columns and "llm_model" in df.columns:
        join_keys = ["keyword", "url", "search_engine", "llm_model"]
        right = p[join_keys].drop_duplicates()
    else:
        right = p[["keyword", "url"]].drop_duplicates()
    return df.merge(right, on=join_keys, how="inner")


def short_model_name(model: str) -> str:
    """Strip the org prefix: 'meta-llama/Llama-3.3-70B-Instruct' -> 'Llama-3.3-70B-Instruct'."""
    return model.split("/")[-1] if "/" in model else model


def engine_from_serp_backend(backend: str) -> str:
    """Map the CLI --serp-backend convention to main-table search_engine values."""
    return "duckduckgo" if backend == "ddg" else backend


__all__ = [
    "EXPECTED_COUNTS",
    "RUN_IDS",
    "engine_from_serp_backend",
    "filter_to_robust",
    "load_robust_winner_pairs",
    "short_model_name",
]
