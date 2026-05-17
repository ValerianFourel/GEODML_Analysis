"""Stage D - DoubleML PLR/IRM grid over (subset x outcome x treatment x method x learner).

Inputs:
    geodml_data/data/main/full_experiment_data_{variant}.parquet  (from merge.py)

Outputs:
    geodml_data/data/dml_results/dml_results_long_{variant}.parquet
    geodml_data/data/dml_results/.dml_{variant}_ckpt.json

Schema of the long table (one row per fit):
    variant, subset, outcome, treatment, method, learner,
    n_obs, coef, se, t_stat, p_val, ci_lower, ci_upper, sig_stars,
    interpretation, error

Subsets:
    POOLED         - everything
    by_engine      - one row per (engine,)
    by_model       - one row per (llm_model,)
    by_pool        - one row per (pool,)
    by_engine_model_pool - all (engine, llm_model, pool) cells

Ported verbatim from ``pipeline/analyze.py`` (preprocess, _get_learners,
run_dml, run_ols, significance_stars, interpret) with two changes:

- CLI consumes the variant-suffixed parquet from ``pipeline.merge``.
- Plotting helpers from the original ``analyze.py`` are dropped;
  ``interpretability.make_figures`` is the canonical plot driver.

Pure CPU; no GPU; safe for sbatch CPU-only or login-node runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from interpretability.pipeline import config as C
from interpretability.utils import Checkpoint, data_root

warnings.filterwarnings("ignore", category=UserWarning)


# ── Subset definitions ───────────────────────────────────────────────────────

def _iter_subsets(df: pd.DataFrame, requested: list[str]):
    """Yield (subset_label, sub_df). Slices the parquet into the requested cells."""
    if "POOLED" in requested:
        yield "POOLED", df

    if "by_engine" in requested and "search_engine" in df.columns:
        for eng, g in df.groupby("search_engine"):
            yield f"by_engine={eng}", g

    if "by_model" in requested and "llm_model" in df.columns:
        for m, g in df.groupby("llm_model"):
            yield f"by_model={m}", g

    if "by_pool" in requested and "pool" in df.columns:
        for p, g in df.groupby("pool"):
            yield f"by_pool={int(p)}", g

    if "by_engine_model_pool" in requested and all(
        c in df.columns for c in ("search_engine", "llm_model", "pool")
    ):
        for (eng, m, p), g in df.groupby(["search_engine", "llm_model", "pool"]):
            yield f"by_engine_model_pool=({eng},{m},{int(p)})", g


# ── Preprocessing (verbatim from analyze.py with sklearn imports lazy) ───────

def preprocess(df: pd.DataFrame, treatment_col: str, outcome_col: str,
               confounders: list[str]):
    """Return X_scaled, Y, D, n_after, missing_info.

    Drops rows with missing treatment or outcome; median-imputes confounders;
    z-standardizes confounders. Returns (None, None, None, n_after, info)
    if too few observations remain.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    cols_needed = list(confounders) + [treatment_col, outcome_col]
    sub = df[cols_needed].copy()

    miss = sub.isna().sum()
    miss_pct = (miss / len(sub) * 100).round(1)
    missing_info = {
        c: f"{int(miss[c])}/{len(sub)} ({miss_pct[c]}%)"
        for c in cols_needed if miss[c] > 0
    }

    sub = sub.dropna(subset=[treatment_col, outcome_col])
    n_after = len(sub)
    if n_after < 10:
        return None, None, None, n_after, missing_info

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(sub[confounders]),
        columns=list(confounders), index=sub.index,
    )

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed),
        columns=list(confounders), index=sub.index,
    )

    Y = sub[outcome_col].values
    D = sub[treatment_col].values
    return X_scaled, Y, D, n_after, missing_info


# ── Learners (verbatim from analyze.py) ──────────────────────────────────────

def _get_learners(learner_type: str, method: str):
    if learner_type == "lgbm":
        from lightgbm import LGBMRegressor, LGBMClassifier
        ml_l = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                             num_leaves=31, verbose=-1, random_state=42)
        ml_m = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                             num_leaves=31, verbose=-1, random_state=42)
        if method == "irm":
            ml_m = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                                  num_leaves=31, verbose=-1, random_state=42)
    elif learner_type == "rf":
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        ml_l = RandomForestRegressor(n_estimators=200, max_depth=5,
                                     random_state=42, n_jobs=-1)
        ml_m = RandomForestRegressor(n_estimators=200, max_depth=5,
                                     random_state=42, n_jobs=-1)
        if method == "irm":
            ml_m = RandomForestClassifier(n_estimators=200, max_depth=5,
                                          random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"learner_type must be lgbm or rf, got {learner_type!r}")
    return ml_l, ml_m


# ── DML / OLS (verbatim from analyze.py) ─────────────────────────────────────

def run_dml(X: pd.DataFrame, Y: np.ndarray, D: np.ndarray,
            method: str = "plr", learner_type: str = "lgbm",
            n_folds: int = 5) -> dict:
    import doubleml as dml
    ml_l, ml_m = _get_learners(learner_type, method)

    dml_data = dml.DoubleMLData.from_arrays(x=X.values, y=Y, d=D)

    if method == "plr":
        model = dml.DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m,
                                n_folds=n_folds, score="partialling out")
    elif method == "irm":
        if len(np.unique(D)) > 2:
            median_d = float(np.median(D))
            D_bin = (D > median_d).astype(float)
            dml_data = dml.DoubleMLData.from_arrays(x=X.values, y=Y, d=D_bin)
        model = dml.DoubleMLIRM(dml_data, ml_g=ml_l, ml_m=ml_m,
                                n_folds=n_folds, score="ATE")
    else:
        raise ValueError(f"unknown method: {method}")

    model.fit()

    coef = float(model.coef[0])
    se = float(model.se[0])
    t_stat = float(model.t_stat[0])
    p_val = float(model.pval[0])
    ci = model.confint(level=0.95)
    return {
        "coef":     coef,
        "se":       se,
        "t_stat":   t_stat,
        "p_val":    p_val,
        "ci_lower": float(ci.iloc[0, 0]),
        "ci_upper": float(ci.iloc[0, 1]),
    }


def run_ols(X: pd.DataFrame, Y: np.ndarray, D: np.ndarray) -> dict:
    from scipy import stats
    n = len(Y)
    design = np.column_stack([np.ones(n), D, X.values])
    coeffs, _, _, _ = np.linalg.lstsq(design, Y, rcond=None)
    beta = float(coeffs[1])
    residuals = Y - design @ coeffs
    sigma2 = float(np.sum(residuals ** 2) / (n - design.shape[1]))
    cov = sigma2 * np.linalg.inv(design.T @ design)
    se = float(np.sqrt(cov[1, 1]))
    if se > 0:
        t_stat = beta / se
        p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - design.shape[1])))
    else:
        t_stat, p_val = 0.0, 1.0
    return {
        "coef":     beta,
        "se":       se,
        "t_stat":   float(t_stat),
        "p_val":    p_val,
        "ci_lower": beta - 1.96 * se,
        "ci_upper": beta + 1.96 * se,
    }


def significance_stars(p: float | None) -> str:
    if p is None or pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def interpret(treatment_name: str, coef: float | None, p_val: float | None,
              outcome: str = "rank_delta") -> str:
    if coef is None or p_val is None:
        return "(no estimate)"
    label = C.TREATMENT_LABELS.get(treatment_name, treatment_name)
    sig = "significantly " if p_val < 0.05 else "not significantly "
    magnitude = abs(coef)
    if outcome == "rank_delta":
        direction = "improves LLM re-ranking" if coef > 0 else "worsens LLM re-ranking"
        return f"{label} {sig}{direction} by {magnitude:.2f} positions (p={p_val:.3f})"
    direction = "improves" if coef < 0 else "worsens"
    return f"{label} {sig}{direction} {outcome} by {magnitude:.2f} positions (p={p_val:.3f})"


# ── Driver ───────────────────────────────────────────────────────────────────

def _select_treatments(measurement: str) -> dict[str, str]:
    if measurement == "all":
        return {**C.TREATMENTS_CODE, **C.TREATMENTS_LLM, **C.TREATMENTS_NEW}
    if measurement == "code":
        return dict(C.TREATMENTS_CODE)
    if measurement == "llm":
        return dict(C.TREATMENTS_LLM)
    if measurement == "new":
        return dict(C.TREATMENTS_NEW)
    raise ValueError(f"unknown measurement: {measurement!r}")


def _filter_available(df: pd.DataFrame, treatments: dict[str, str],
                      confounders: list[str]) -> tuple[dict[str, str], list[str]]:
    avail_t: dict[str, str] = {}
    for name, col in treatments.items():
        if col not in df.columns or df[col].notna().sum() == 0:
            continue
        if df[col].dropna().nunique() < 2:
            continue
        avail_t[name] = col

    avail_c = [c for c in confounders
               if c in df.columns
               and df[c].notna().sum() > 0
               and df[c].dropna().nunique() > 1]
    return avail_t, avail_c


def _ckpt_key(variant: str, subset: str, outcome: str, treatment: str,
              method: str, learner: str) -> str:
    return f"{variant}::{subset}::{outcome}::{treatment}::{method}::{learner}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage D: DoubleML PLR/IRM grid over the merged main table.",
    )
    ap.add_argument("--variant", required=True,
                    choices=(
                        "biased", "neutral",
                        "biased_passage", "neutral_passage",
                        "biased_rag", "neutral_rag",
                    ))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--input", default=None,
                    help="Override input parquet (default: data/main/full_experiment_data_{variant}.parquet).")
    ap.add_argument("--output", default=None,
                    help="Override output parquet (default: data/dml_results/dml_results_long_{variant}.parquet).")
    ap.add_argument("--measurement", default="new",
                    choices=("code", "llm", "new", "all"),
                    help="Which treatment family. Default 'new' (T1a..T7).")
    ap.add_argument("--treatments", nargs="*", default=None,
                    help="Explicit short-label list (overrides --measurement).")
    ap.add_argument("--outcomes", nargs="*",
                    default=("rank_delta", "post_rank"),
                    help="Outcomes to fit. Default: rank_delta + post_rank.")
    ap.add_argument("--methods", nargs="*", default=("plr",),
                    choices=("plr", "irm"))
    ap.add_argument("--learners", nargs="*", default=("lgbm", "rf"),
                    choices=("lgbm", "rf"))
    ap.add_argument("--subsets", nargs="*",
                    default=("POOLED", "by_engine", "by_model", "by_pool"),
                    help="Which slices of the data to fit. Default: POOLED + by_engine + by_model + by_pool.")
    ap.add_argument("--n-folds", type=int, default=C.DML_N_FOLDS)
    ap.add_argument("--ols", action="store_true",
                    help="Also emit naive OLS rows (method=ols) for comparison.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip fits already present in the output parquet / checkpoint.")
    args = ap.parse_args()

    root = data_root(args.data_root)
    in_path  = Path(args.input) if args.input else C.main_table_path(args.variant, root)
    out_path = Path(args.output) if args.output else C.dml_results_path(args.variant, root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[dml] FATAL: input not found: {in_path}", file=sys.stderr)
        return 2

    df = pd.read_parquet(in_path)
    print(f"[dml] loaded {len(df):,} rows ({df.shape[1]} cols) <- {in_path}", flush=True)

    if args.treatments:
        treatments = {t: C.ALL_TREATMENTS[t] for t in args.treatments
                      if t in C.ALL_TREATMENTS}
    else:
        treatments = _select_treatments(args.measurement)

    avail_t, avail_c = _filter_available(df, treatments, C.CONFOUNDERS)
    if not avail_t:
        print("[dml] FATAL: no treatment columns available with variance",
              file=sys.stderr)
        return 2
    if not avail_c:
        print("[dml] WARN: zero confounders available; DML will reduce to univariate",
              flush=True)

    print(f"[dml] treatments    : {list(avail_t)}", flush=True)
    print(f"[dml] confounders   : {avail_c}", flush=True)
    print(f"[dml] outcomes      : {list(args.outcomes)}", flush=True)
    print(f"[dml] methods       : {list(args.methods)}{'  (+ ols)' if args.ols else ''}",
          flush=True)
    print(f"[dml] learners      : {list(args.learners)}", flush=True)
    print(f"[dml] subsets       : {list(args.subsets)}", flush=True)
    print(f"[dml] n_folds       : {args.n_folds}", flush=True)

    ckpt = Checkpoint.load(
        root / "data" / "dml_results" / f".dml_{args.variant}_ckpt.json"
    )
    seen = set(ckpt.data.get("seen", []))

    # Read existing rows so resume keeps them.
    existing_rows: list[dict] = []
    if args.resume and out_path.exists():
        existing_rows = pd.read_parquet(out_path).to_dict(orient="records")
        print(f"[dml] resuming: {len(existing_rows):,} existing rows", flush=True)
    elif out_path.exists():
        out_path.unlink()

    all_rows: list[dict] = list(existing_rows)
    n_fit = 0
    n_skipped = 0
    n_err = 0
    total = (
        sum(1 for _ in _iter_subsets(df, list(args.subsets)))
        * len(avail_t) * len(args.outcomes) * len(args.methods) * len(args.learners)
    )
    print(f"[dml] grid size     : ~{total} fits", flush=True)
    t0 = time.time()

    for subset_label, sub_df in _iter_subsets(df, list(args.subsets)):
        for outcome in args.outcomes:
            if outcome not in sub_df.columns:
                continue
            for treatment_name, treatment_col in avail_t.items():
                for method in args.methods:
                    for learner in args.learners:
                        key = _ckpt_key(args.variant, subset_label, outcome,
                                        treatment_name, method, learner)
                        if args.resume and key in seen:
                            n_skipped += 1
                            continue

                        X, Y, D, n_obs, _ = preprocess(
                            sub_df, treatment_col, outcome, avail_c,
                        )

                        row_base = {
                            "variant":   args.variant,
                            "subset":    subset_label,
                            "outcome":   outcome,
                            "treatment": treatment_name,
                            "method":    method,
                            "learner":   learner,
                            "n_obs":     n_obs,
                        }

                        if X is None:
                            row = {**row_base, "coef": None, "se": None,
                                   "t_stat": None, "p_val": None,
                                   "ci_lower": None, "ci_upper": None,
                                   "sig_stars": "", "error": f"too_few_obs ({n_obs})",
                                   "interpretation": ""}
                            all_rows.append(row)
                            ckpt.mark(key)
                            continue

                        try:
                            res = run_dml(X, Y, D, method=method,
                                          learner_type=learner, n_folds=args.n_folds)
                            err = None
                        except Exception as e:
                            res = {"coef": None, "se": None, "t_stat": None,
                                   "p_val": None, "ci_lower": None, "ci_upper": None}
                            err = f"{type(e).__name__}: {e}"
                            n_err += 1
                            if n_err <= 3:
                                traceback.print_exc()

                        row = {
                            **row_base, **res,
                            "sig_stars": significance_stars(res["p_val"]),
                            "interpretation": interpret(
                                treatment_name, res["coef"], res["p_val"], outcome,
                            ),
                            "error": err,
                        }
                        all_rows.append(row)
                        ckpt.mark(key)
                        n_fit += 1
                        if n_fit % 10 == 0:
                            elapsed = time.time() - t0
                            print(f"[dml]   {n_fit} fits done ({elapsed:.1f}s)",
                                  flush=True)
                            pd.DataFrame(all_rows).to_parquet(out_path, index=False)
                            ckpt.save()

                        # Optional naive-OLS row (only on plr+lgbm to avoid dups)
                        if args.ols and method == "plr" and learner == "lgbm":
                            ols_key = _ckpt_key(args.variant, subset_label, outcome,
                                                treatment_name, "ols", "ols")
                            if not (args.resume and ols_key in seen):
                                try:
                                    ols_res = run_ols(X, Y, D)
                                except Exception as e:
                                    ols_res = {"coef": None, "se": None,
                                               "t_stat": None, "p_val": None,
                                               "ci_lower": None, "ci_upper": None}
                                    ols_err = f"{type(e).__name__}: {e}"
                                else:
                                    ols_err = None
                                ols_row = {
                                    "variant":   args.variant,
                                    "subset":    subset_label,
                                    "outcome":   outcome,
                                    "treatment": treatment_name,
                                    "method":    "ols",
                                    "learner":   "ols",
                                    "n_obs":     n_obs,
                                    **ols_res,
                                    "sig_stars": significance_stars(ols_res.get("p_val")),
                                    "interpretation": interpret(
                                        treatment_name, ols_res.get("coef"),
                                        ols_res.get("p_val"), outcome,
                                    ),
                                    "error": ols_err,
                                }
                                all_rows.append(ols_row)
                                ckpt.mark(ols_key)

    pd.DataFrame(all_rows).to_parquet(out_path, index=False)
    ckpt.save()

    print(f"\n[dml] done: new={n_fit} skipped={n_skipped} errors={n_err} "
          f"total_rows={len(all_rows)} -> {out_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
