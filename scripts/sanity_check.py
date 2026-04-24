"""Confirm the POOLED DML coefficients on rank_delta match the paper table."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


EXPECTED_SIGNS = {
    "T7_source_earned": "-",         # strong demoter
    "T5_topical_comp": "+",          # strong promoter
    "T3_structured_data_new": "-",
    "T6_freshness": "-",
    "T2a_question_headings": "+",
}


def main() -> int:
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root", default=os.getenv("GEODML_DATA_ROOT", "./geodml_data")
    )
    args = ap.parse_args()

    root = Path(args.data_root)
    f = root / "data" / "dml_results" / "dml_results_long.parquet"
    if not f.exists():
        print(f"ERROR: {f} not found. Run scripts/download_data.py first.")
        return 2

    fits = pd.read_parquet(f)
    pooled = fits[(fits.subset == "POOLED") & (fits.outcome == "rank_delta")]
    sig = pooled[pooled.p_val < 0.01].sort_values("coef")
    print(f"{len(sig)} significant POOLED fits on rank_delta\n")
    print(sig[["treatment", "coef", "se", "p_val", "stars"]].to_string(index=False))

    print("\nSign check:")
    ok = True
    for t, want in EXPECTED_SIGNS.items():
        row = pooled[pooled.treatment == t]
        if row.empty:
            print(f"  [?] {t:<30s} not found")
            ok = False
            continue
        got = "+" if row.coef.iloc[0] > 0 else "-"
        flag = "OK" if got == want else "MISMATCH"
        print(
            f"  [{flag}] {t:<30s} coef={row.coef.iloc[0]:+.3f} (expected {want})"
        )
        ok = ok and got == want
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
