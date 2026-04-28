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


FRAME_SUFFIX = {"full": "_full", "robust_winners": "_rw"}


def _load_dml_variant(root: Path, variant: str) -> pd.DataFrame | None:
    """Try variant-suffixed parquet first; fall back to legacy un-suffixed for biased."""
    p = root / "data" / "dml_results" / f"dml_results_long_{variant}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    if variant == "biased":
        legacy = root / "data" / "dml_results" / "dml_results_long.parquet"
        if legacy.exists():
            return pd.read_parquet(legacy)
    return None


def _resolve_frame_inputs(output_dir: Path, stem: str) -> dict[str, Path]:
    """Return {frame: path} for any frame-suffixed CSVs that exist."""
    found: dict[str, Path] = {}
    for frame, suf in FRAME_SUFFIX.items():
        p = output_dir / f"{stem}{suf}.csv"
        if p.exists():
            found[frame] = p
    if not found:
        legacy = output_dir / f"{stem}.csv"
        if legacy.exists():
            found["full"] = legacy
    return found


def figure_a(output_dir: Path, data_root_path: Path) -> list[Path]:
    found = _resolve_frame_inputs(output_dir, "ablation_results")
    if not found:
        print(f"[fig A] no ablation_results*.csv in {output_dir}, skipping")
        return []

    dml = load_dml_long(data_root_path)
    dml = dml[(dml.subset == "POOLED") & (dml.outcome == "rank_delta")]
    dml_coefs = dml.set_index("treatment")["coef"].to_dict()

    outs: list[Path] = []
    for frame, abl_path in found.items():
        abl = pd.read_csv(abl_path)
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
        ax.set_title(f"Figure A — DML vs direct input ablation [{frame}]")
        ax.legend(loc="best", fontsize=9)
        out = output_dir / "plots" / f"figure_a_ablation{FRAME_SUFFIX[frame]}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"[fig A] -> {out}  (frame={frame}, n_treatments={len(agg)})")
        outs.append(out)
    return outs


def figure_b(output_dir: Path) -> list[Path]:
    found = _resolve_frame_inputs(output_dir, "saliency_summary")
    if not found:
        print(f"[fig B] no saliency_summary*.csv in {output_dir}, skipping")
        return []

    outs: list[Path] = []
    for frame, summary_path in found.items():
        summary = pd.read_csv(summary_path)
        fig, ax = plt.subplots(figsize=(8, 4.5))

        piv = summary.set_index("treatment")[
            ["mean_treatment_saliency", "mean_other_saliency"]
        ]
        piv.columns = ["treatment tokens", "other tokens"]
        sns.heatmap(piv.T, annot=True, fmt=".3f", cmap="mako", ax=ax, cbar=True)
        ax.set_title(f"Figure B — Mean gradient×input saliency [{frame}]")
        out = output_dir / "plots" / f"figure_b_saliency{FRAME_SUFFIX[frame]}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"[fig B] -> {out}  (frame={frame})")
        outs.append(out)
    return outs


def figure_c(output_dir: Path) -> Path:
    probing_path = output_dir / "probing_results.csv"
    if not probing_path.exists():
        print(f"[fig C] missing {probing_path}, skipping")
        return Path()
    df = pd.read_csv(probing_path)
    has_frame = "frame" in df.columns and df["frame"].nunique() > 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    treatments = sorted(df["treatment"].unique())
    cmap = plt.get_cmap("tab10")
    treat_colors = {t: cmap(i % 10) for i, t in enumerate(treatments)}
    frame_styles = {"full": "--", "robust_winners": "-"}
    for ax, pooling in zip(axes, ["last_token", "mean"]):
        sub = df[df.pooling == pooling]
        if has_frame:
            for (t, fr), g in sub.groupby(["treatment", "frame"]):
                g = g.sort_values("layer")
                ax.plot(
                    g["layer"], g["accuracy"],
                    marker="o", markersize=3, lw=1.2,
                    color=treat_colors[t],
                    linestyle=frame_styles.get(fr, "-"),
                    label=f"{t} [{fr}]",
                )
        else:
            for t, g in sub.groupby("treatment"):
                g = g.sort_values("layer")
                ax.plot(g["layer"], g["accuracy"], marker="o", lw=1.2,
                        color=treat_colors[t], label=t)
        ax.axhline(0.5, ls=":", color="grey", lw=0.6, label="chance")
        ax.set_xlabel("layer")
        ax.set_title(f"pooling = {pooling}")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("probe test accuracy")
    axes[0].legend(loc="best", fontsize=7, ncol=2 if has_frame else 1)
    suffix = "frames" if has_frame else "single"
    fig.suptitle(
        f"Figure C — Probing accuracy by layer "
        f"({'full vs robust_winners' if has_frame else 'single frame'})"
    )
    out = output_dir / "plots" / "figure_c_probing.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[fig C] -> {out}  ({suffix})")
    return out


def figure_d(output_dir: Path) -> list[Path]:
    """Two panels: logit-lens curve (left), head-importance heatmap (right)."""
    lens_path = output_dir / "logit_lens.csv"
    heads_path = output_dir / "attention_heads.csv"
    outs: list[Path] = []

    if lens_path.exists():
        lens = pd.read_csv(lens_path)
        curve = (lens.groupby("layer")["domain_token_prob_mass"]
                 .agg(["mean", "sem"]).reset_index())
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(curve["layer"], curve["mean"], marker="o", lw=1.4, color="darkviolet")
        ax.fill_between(
            curve["layer"],
            curve["mean"] - curve["sem"],
            curve["mean"] + curve["sem"],
            alpha=0.2, color="darkviolet",
        )
        ax.set_xlabel("layer")
        ax.set_ylabel("P(domain-like token | decision position)")
        ax.set_title("Figure D1 — Logit lens: when does the ranker decide on a domain?")
        ax.grid(alpha=0.25)
        out = output_dir / "plots" / "figure_d1_logit_lens.png"
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        outs.append(out)
        print(f"[fig D1] -> {out}")

    if heads_path.exists():
        heads = pd.read_csv(heads_path)
        piv = (heads.groupby(["layer", "head"])["attn_to_url"].mean()
               .unstack("head"))
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(piv, cmap="magma", ax=ax, cbar_kws={"label": "mean attn to URL"})
        ax.set_title("Figure D2 — Attention head importance for URL tokens")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        out = output_dir / "plots" / "figure_d2_head_importance.png"
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        outs.append(out)
        print(f"[fig D2] -> {out}")

    return outs


def figure_a_compare(output_dir: Path, data_root_path: Path) -> list[Path]:
    """Side-by-side biased vs neutral DML coefficients + delta plot.

    Reads ``dml_results_long_{biased,neutral}.parquet`` if present and emits:
        plots/figure_a_dml_biased.png   - one subplot per outcome (POOLED subset)
        plots/figure_a_dml_neutral.png  - one subplot per outcome (POOLED subset)
        plots/figure_a_dml_delta.png    - per-treatment Δcoef = neutral − biased

    No-op (with a print) if neither variant parquet exists. If only one exists,
    emits that single panel and skips the delta plot.
    """
    df_b = _load_dml_variant(data_root_path, "biased")
    df_n = _load_dml_variant(data_root_path, "neutral")
    if df_b is None and df_n is None:
        print(f"[fig A_compare] no dml_results_long_{{biased,neutral}}.parquet found "
              f"under {data_root_path}/data/dml_results/, skipping")
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    outs: list[Path] = []

    def _draw_single(df: pd.DataFrame, title: str, out_path: Path) -> None:
        # POOLED subset, default learner=lgbm, method=plr.
        sub = df[(df["subset"] == "POOLED")
                 & (df["method"] == "plr")
                 & (df["learner"] == "lgbm")
                 & df["coef"].notna()]
        if sub.empty:
            print(f"[fig A_compare] {title}: no POOLED+plr+lgbm rows, skipping")
            return
        outcomes = sorted(sub["outcome"].unique())
        fig, axes = plt.subplots(1, len(outcomes), figsize=(5.5 * len(outcomes), 5.5),
                                 sharey=True)
        if len(outcomes) == 1:
            axes = [axes]
        for ax, outcome in zip(axes, outcomes):
            s = sub[sub.outcome == outcome].sort_values("coef")
            ax.errorbar(
                s["coef"], range(len(s)),
                xerr=1.96 * s["se"],
                fmt="o", color="steelblue", capsize=3,
            )
            ax.set_yticks(range(len(s)))
            ax.set_yticklabels(s["treatment"], fontsize=8)
            ax.axvline(0, color="grey", lw=0.5)
            ax.set_xlabel(f"DML coefficient on {outcome}")
            ax.set_title(outcome)
            ax.grid(alpha=0.25)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[fig A_compare] -> {out_path}")
        outs.append(out_path)

    if df_b is not None:
        _draw_single(df_b, "Figure A - DML coefficients [biased prompt]",
                     plots_dir / "figure_a_dml_biased.png")
    if df_n is not None:
        _draw_single(df_n, "Figure A - DML coefficients [neutral prompt]",
                     plots_dir / "figure_a_dml_neutral.png")

    # Delta plot only if both variants exist.
    if df_b is not None and df_n is not None:
        keys = ["subset", "outcome", "treatment", "method", "learner"]
        join = df_b.merge(df_n, on=keys, suffixes=("_b", "_n"))
        join = join[(join["subset"] == "POOLED")
                    & (join["method"] == "plr")
                    & (join["learner"] == "lgbm")
                    & join["coef_b"].notna()
                    & join["coef_n"].notna()]
        if join.empty:
            print("[fig A_compare] no overlapping POOLED+plr+lgbm rows for delta plot")
            return outs

        join["delta"] = join["coef_n"] - join["coef_b"]
        # Combined SE under independence: sqrt(se_b^2 + se_n^2).
        join["delta_se"] = np.sqrt(join["se_b"]**2 + join["se_n"]**2)
        outcomes = sorted(join["outcome"].unique())

        fig, axes = plt.subplots(1, len(outcomes), figsize=(5.5 * len(outcomes), 5.5),
                                 sharey=True)
        if len(outcomes) == 1:
            axes = [axes]
        for ax, outcome in zip(axes, outcomes):
            s = join[join.outcome == outcome].sort_values("delta")
            ax.errorbar(
                s["delta"], range(len(s)),
                xerr=1.96 * s["delta_se"],
                fmt="o", color="crimson", capsize=3,
            )
            ax.set_yticks(range(len(s)))
            ax.set_yticklabels(s["treatment"], fontsize=8)
            ax.axvline(0, color="grey", lw=0.5)
            ax.set_xlabel(f"Δcoef = neutral − biased  ({outcome})")
            ax.set_title(outcome)
            ax.grid(alpha=0.25)
        fig.suptitle("Figure A delta - prompt-bias attributable shift in DML coefficients")
        fig.tight_layout()
        out = plots_dir / "figure_a_dml_delta.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"[fig A_compare] -> {out}")
        outs.append(out)

    return outs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "output"),
    )
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--skip-compare", action="store_true",
                    help="Skip figure_a_compare (biased vs neutral) even if both exist.")
    args = ap.parse_args()

    out = Path(args.output_dir)
    root = data_root(args.data_root)
    figure_a(out, root)
    figure_b(out)
    figure_c(out)
    figure_d(out)
    if not args.skip_compare:
        figure_a_compare(out, root)
    print("[make_figures] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
