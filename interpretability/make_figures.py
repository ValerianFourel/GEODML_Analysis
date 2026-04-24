"""Produce the three paper figures from CSV outputs.

Figure A — Ablation vs DML: per-treatment DML coef (x) vs mean ablation_delta
           (y). Perfect validation lies on y = -x (ablating a promoter lowers
           the rank).
Figure B — T7 saliency heatmap: mean saliency per token bucket for earned vs
           brand pages (collapsed across keywords).
Figure C — Probing curves: probe accuracy by layer, one line per treatment,
           two panels (last_token and mean pooling).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from interpretability.utils import data_root, load_dml_long

load_dotenv()


def figure_a(output_dir: Path, data_root_path: Path) -> Path:
    abl_path = output_dir / "ablation_results.csv"
    if not abl_path.exists():
        print(f"[fig A] missing {abl_path}, skipping")
        return Path()

    abl = pd.read_csv(abl_path)
    dml = load_dml_long(data_root_path)
    dml = dml[(dml.subset == "POOLED") & (dml.outcome == "rank_delta")]
    dml_coefs = dml.set_index("treatment")["coef"].to_dict()

    agg = (
        abl.groupby("treatment")["ablation_delta"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    agg["dml_coef"] = agg["treatment"].map(dml_coefs)
    agg = agg.dropna(subset=["dml_coef"])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)

    if not agg.empty:
        xs = agg["dml_coef"].to_numpy()
        ref_x = np.linspace(xs.min() * 1.1, xs.max() * 1.1, 50)
        ax.plot(ref_x, -ref_x, "--", color="crimson", alpha=0.6,
                label="Perfect agreement: y = -x")

    ax.errorbar(
        agg["dml_coef"], agg["mean"], yerr=agg["sem"],
        fmt="o", color="steelblue", capsize=3, label="treatment",
    )
    for _, r in agg.iterrows():
        ax.annotate(
            r["treatment"], (r["dml_coef"], r["mean"]),
            xytext=(4, 4), textcoords="offset points", fontsize=8,
        )
    ax.set_xlabel("DML coefficient on rank_delta (higher = promoter)")
    ax.set_ylabel("Mean ablation_delta (higher = ablation hurt the page)")
    ax.set_title("Figure A — DML estimates vs direct input ablation")
    ax.legend(loc="best", fontsize=9)
    out = output_dir / "plots" / "figure_a_ablation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[fig A] -> {out}  (n_treatments={len(agg)})")
    return out


def figure_b(output_dir: Path) -> Path:
    scores_path = output_dir / "saliency_scores.csv"
    summary_path = output_dir / "saliency_summary.csv"
    if not summary_path.exists():
        print(f"[fig B] missing {summary_path}, skipping")
        return Path()

    summary = pd.read_csv(summary_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    piv = summary.set_index("treatment")[
        ["mean_treatment_saliency", "mean_other_saliency"]
    ]
    piv.columns = ["treatment tokens", "other tokens"]
    sns.heatmap(piv.T, annot=True, fmt=".3f", cmap="mako", ax=ax, cbar=True)
    ax.set_title("Figure B — Mean gradient×input saliency per treatment")
    out = output_dir / "plots" / "figure_b_saliency.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[fig B] -> {out}")
    return out


def figure_c(output_dir: Path) -> Path:
    probing_path = output_dir / "probing_results.csv"
    if not probing_path.exists():
        print(f"[fig C] missing {probing_path}, skipping")
        return Path()
    df = pd.read_csv(probing_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, pooling in zip(axes, ["last_token", "mean"]):
        sub = df[df.pooling == pooling]
        for t, g in sub.groupby("treatment"):
            g = g.sort_values("layer")
            ax.plot(g["layer"], g["accuracy"], marker="o", lw=1.2, label=t)
        ax.axhline(0.5, ls="--", color="grey", lw=0.6, label="chance")
        ax.set_xlabel("layer")
        ax.set_title(f"pooling = {pooling}")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("probe test accuracy")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Figure C — Probing accuracy by layer, per treatment")
    out = output_dir / "plots" / "figure_c_probing.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[fig C] -> {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "output"),
    )
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    out = Path(args.output_dir)
    root = data_root(args.data_root)
    figure_a(out, root)
    figure_b(out)
    figure_c(out)
    print("[make_figures] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
