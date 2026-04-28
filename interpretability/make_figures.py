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
KNOWN_VARIANTS = ("biased", "neutral")


def _resolve_variant_csv(output_dir: Path, stem: str) -> dict[str, Path]:
    """Find variant-suffixed merged CSVs, falling back to the un-suffixed
    legacy file as the 'biased' variant.

    Returns ``{variant: path}`` for whichever variants have a CSV present.
    """
    found: dict[str, Path] = {}
    for v in KNOWN_VARIANTS:
        p = output_dir / f"{stem}_{v}.csv"
        if p.exists():
            found[v] = p
    if "biased" not in found:
        legacy = output_dir / f"{stem}.csv"
        if legacy.exists():
            found["biased"] = legacy
    return found


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


def _resolve_frame_inputs(output_dir: Path, stem: str,
                          variant: str | None = None) -> dict[str, Path]:
    """Return {frame: path} for any frame-suffixed CSVs that exist.

    When ``variant`` is given (e.g. ``"biased"``), looks for
    ``{stem}{frame_suffix}_{variant}.csv`` first and falls back to the
    legacy un-suffixed file. When ``variant`` is None, only the legacy
    layout is searched (preserves pre-port behavior).
    """
    found: dict[str, Path] = {}
    for frame, suf in FRAME_SUFFIX.items():
        candidates: list[Path] = []
        if variant is not None:
            candidates.append(output_dir / f"{stem}{suf}_{variant}.csv")
        candidates.append(output_dir / f"{stem}{suf}.csv")
        for p in candidates:
            if p.exists():
                found[frame] = p
                break
    if not found:
        legacy = output_dir / f"{stem}.csv"
        if legacy.exists():
            found["full"] = legacy
    return found


def figure_a(output_dir: Path, data_root_path: Path) -> list[Path]:
    """Per-variant ablation × DML scatter. Looks for variant-suffixed
    ``ablation_results_{full,rw}_{variant}.csv`` first; falls back to
    legacy un-suffixed paths (treated as 'biased')."""
    outs: list[Path] = []
    # Discover which variants have ablation_results CSVs available.
    discovered: dict[str, dict[str, Path]] = {}
    for v in KNOWN_VARIANTS:
        found_v = _resolve_frame_inputs(output_dir, "ablation_results", variant=v)
        # Filter to variant-suffixed files only when v is known; the helper
        # may have returned a legacy path which we reserve for the un-tagged
        # call below.
        found_v = {fr: p for fr, p in found_v.items()
                   if p.name.endswith(f"_{v}.csv")}
        if found_v:
            discovered[v] = found_v
    # Legacy un-suffixed (only used if no variant-suffixed CSV at all).
    if not discovered:
        legacy = _resolve_frame_inputs(output_dir, "ablation_results")
        if legacy:
            discovered["biased"] = legacy
    if not discovered:
        print(f"[fig A] no ablation_results*.csv in {output_dir}, skipping")
        return []

    for variant, found in discovered.items():
        # DML lookup respects the same variant.
        dml = _load_dml_variant(data_root_path, variant)
        if dml is None:
            print(f"[fig A] no DML parquet for variant={variant}, skipping ablation×DML scatter")
            continue
        dml = dml[(dml.subset == "POOLED") & (dml.outcome == "rank_delta")]
        dml_coefs = dml.set_index("treatment")["coef"].to_dict()

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
            ax.set_title(f"Figure A — DML vs direct input ablation [{frame}, {variant}]")
            ax.legend(loc="best", fontsize=9)
            stem = f"figure_a_ablation{FRAME_SUFFIX[frame]}_{variant}"
            out = output_dir / "plots" / f"{stem}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(out, dpi=200)
            plt.close(fig)
            print(f"[fig A] -> {out}  (frame={frame}, variant={variant}, n_treatments={len(agg)})")
            outs.append(out)
            # For 'biased' also drop a legacy alias so older readers / docs work.
            if variant == "biased":
                legacy = output_dir / "plots" / f"figure_a_ablation{FRAME_SUFFIX[frame]}.png"
                fig.tight_layout()
                # Re-save under legacy name (cheap; we already have the figure closed,
                # so just write the alias as a copy of the canonical file).
                import shutil
                shutil.copyfile(out, legacy)
    return outs


def figure_b(output_dir: Path) -> list[Path]:
    """Per-variant saliency heatmap. Falls back to legacy un-suffixed CSVs."""
    outs: list[Path] = []
    discovered: dict[str, dict[str, Path]] = {}
    for v in KNOWN_VARIANTS:
        f_v = _resolve_frame_inputs(output_dir, "saliency_summary", variant=v)
        f_v = {fr: p for fr, p in f_v.items()
               if p.name.endswith(f"_{v}.csv")}
        if f_v:
            discovered[v] = f_v
    if not discovered:
        legacy = _resolve_frame_inputs(output_dir, "saliency_summary")
        if legacy:
            discovered["biased"] = legacy
    if not discovered:
        print(f"[fig B] no saliency_summary*.csv in {output_dir}, skipping")
        return []

    for variant, found in discovered.items():
        for frame, summary_path in found.items():
            summary = pd.read_csv(summary_path)
            fig, ax = plt.subplots(figsize=(8, 4.5))

            piv = summary.set_index("treatment")[
                ["mean_treatment_saliency", "mean_other_saliency"]
            ]
            piv.columns = ["treatment tokens", "other tokens"]
            sns.heatmap(piv.T, annot=True, fmt=".3f", cmap="mako", ax=ax, cbar=True)
            ax.set_title(f"Figure B — Mean gradient×input saliency [{frame}, {variant}]")
            stem = f"figure_b_saliency{FRAME_SUFFIX[frame]}_{variant}"
            out = output_dir / "plots" / f"{stem}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(out, dpi=200)
            plt.close(fig)
            print(f"[fig B] -> {out}  (frame={frame}, variant={variant})")
            outs.append(out)
            if variant == "biased":
                import shutil
                shutil.copyfile(
                    out,
                    output_dir / "plots" / f"figure_b_saliency{FRAME_SUFFIX[frame]}.png",
                )
    return outs


def figure_c(output_dir: Path) -> list[Path]:
    """Per-variant probing curves."""
    paths = _resolve_variant_csv(output_dir, "probing_results")
    if not paths:
        print(f"[fig C] missing probing_results*.csv in {output_dir}, skipping")
        return []
    outs: list[Path] = []
    for variant, probing_path in paths.items():
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
            f"({'full vs robust_winners' if has_frame else 'single frame'})  [{variant}]"
        )
        out = output_dir / "plots" / f"figure_c_probing_{variant}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"[fig C] -> {out}  ({suffix}, variant={variant})")
        outs.append(out)
        if variant == "biased":
            import shutil
            shutil.copyfile(out, output_dir / "plots" / "figure_c_probing.png")
    return outs


def figure_d(output_dir: Path) -> list[Path]:
    """Per-variant logit-lens curve + head-importance heatmap."""
    lens_paths = _resolve_variant_csv(output_dir, "logit_lens")
    heads_paths = _resolve_variant_csv(output_dir, "attention_heads")
    outs: list[Path] = []

    for variant, lens_path in lens_paths.items():
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
        ax.set_title(f"Figure D1 — Logit lens: when does the ranker decide on a domain? [{variant}]")
        ax.grid(alpha=0.25)
        out = output_dir / "plots" / f"figure_d1_logit_lens_{variant}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        outs.append(out)
        print(f"[fig D1] -> {out}  (variant={variant})")
        if variant == "biased":
            import shutil
            shutil.copyfile(out, output_dir / "plots" / "figure_d1_logit_lens.png")

    for variant, heads_path in heads_paths.items():
        heads = pd.read_csv(heads_path)
        piv = (heads.groupby(["layer", "head"])["attn_to_url"].mean()
               .unstack("head"))
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(piv, cmap="magma", ax=ax, cbar_kws={"label": "mean attn to URL"})
        ax.set_title(f"Figure D2 — Attention head importance for URL tokens [{variant}]")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        out = output_dir / "plots" / f"figure_d2_head_importance_{variant}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        outs.append(out)
        print(f"[fig D2] -> {out}  (variant={variant})")
        if variant == "biased":
            import shutil
            shutil.copyfile(out, output_dir / "plots" / "figure_d2_head_importance.png")

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


def figure_e_order_sensitivity(output_dir: Path, data_root_path: Path) -> list[Path]:
    """Order-sensitivity probe: overlap of LLM top-K under permuted input.

    Reads ``data/order_probe/order_probe_summary.parquet`` and emits:
        plots/figure_e_order_overlap_by_cell.png   - boxplot of overlap@10 per
            (variant, model, engine, pool), one column per ordering_pair.
        plots/figure_e_order_overlap_biased_vs_neutral.png - paired Δ
            (neutral overlap@10 − biased overlap@10) with 95% bootstrap CI,
            per (model, engine, pool).

    No-op (with a print) if the parquet is missing or empty.
    """
    p = data_root_path / "data" / "order_probe" / "order_probe_summary.parquet"
    if not p.exists():
        print(f"[fig E] missing {p}, skipping (run order_probe_analyze first)")
        return []
    df = pd.read_parquet(p)
    if df.empty:
        print(f"[fig E] {p} has 0 rows, skipping")
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    outs: list[Path] = []

    sub = df[df.K == 10].copy()
    if sub.empty:
        print("[fig E] no K=10 rows; skipping")
        return []
    sub["cell"] = sub["model"] + "/" + sub["engine"] + "/p" + sub["pool"].astype(str)

    # ── Panel 1: boxplot per cell, faceted by ordering_pair ──────────────────
    pairs = sorted(sub["ordering_pair"].unique())
    cells = sorted(sub["cell"].unique())
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 + 2.5 * len(pairs), 6),
                             sharey=True)
    if len(pairs) == 1:
        axes = [axes]
    for ax, pair in zip(axes, pairs):
        data = []
        labels = []
        for cell in cells:
            for variant in ("biased", "neutral"):
                vals = sub[(sub.cell == cell) & (sub.ordering_pair == pair)
                           & (sub.variant == variant)]["overlap_at_k"].values
                if len(vals) == 0:
                    continue
                data.append(vals)
                labels.append(f"{variant[0]}|{cell}")
        if not data:
            ax.set_title(f"{pair} (no data)")
            continue
        bp = ax.boxplot(data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#cfe6ff", color="steelblue"),
                        medianprops=dict(color="darkred"))
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=75, fontsize=7)
        ax.set_title(pair)
        ax.set_ylabel("overlap@10" if ax is axes[0] else "")
        ax.axhline(0.5, color="grey", linestyle="--", lw=0.5, alpha=0.5)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    fig.suptitle("Figure E - LLM rerank top-10 overlap under input permutation")
    fig.tight_layout()
    out1 = plots_dir / "figure_e_order_overlap_by_cell.png"
    fig.savefig(out1, dpi=200)
    plt.close(fig)
    print(f"[fig E] -> {out1}")
    outs.append(out1)

    # ── Panel 2: paired Δ (neutral - biased), per cell × pair ────────────────
    have_both = (sub.groupby(["cell", "ordering_pair", "variant"])["overlap_at_k"]
                    .mean()
                    .unstack("variant"))
    if "biased" in have_both.columns and "neutral" in have_both.columns:
        rng = np.random.default_rng(0)

        def _bootstrap_ci(vals: np.ndarray, n: int = 1000) -> tuple[float, float]:
            if len(vals) < 5:
                return (np.nan, np.nan)
            idx = rng.integers(0, len(vals), size=(n, len(vals)))
            means = vals[idx].mean(axis=1)
            return (float(np.percentile(means, 2.5)),
                    float(np.percentile(means, 97.5)))

        rows = []
        for (cell, pair), grp in sub.groupby(["cell", "ordering_pair"]):
            b = grp[grp.variant == "biased"]["overlap_at_k"].values
            n = grp[grp.variant == "neutral"]["overlap_at_k"].values
            if len(b) == 0 or len(n) == 0:
                continue
            delta = n.mean() - b.mean()
            # Bootstrap on the keyword-level differences (paired by keyword).
            shared = (grp.pivot_table(index="keyword", columns="variant",
                                      values="overlap_at_k", aggfunc="first")
                         .dropna())
            if shared.empty:
                lo, hi = (np.nan, np.nan)
            else:
                diffs = (shared["neutral"] - shared["biased"]).values
                lo, hi = _bootstrap_ci(diffs)
            rows.append({"cell": cell, "ordering_pair": pair,
                         "delta": delta, "ci_lo": lo, "ci_hi": hi,
                         "n_paired": int(len(shared))})
        if rows:
            ddf = pd.DataFrame(rows)
            pairs2 = sorted(ddf["ordering_pair"].unique())
            fig, axes = plt.subplots(1, len(pairs2),
                                     figsize=(5 + 2 * len(pairs2), 5),
                                     sharey=True)
            if len(pairs2) == 1:
                axes = [axes]
            for ax, pair in zip(axes, pairs2):
                d = ddf[ddf.ordering_pair == pair].sort_values("delta")
                ys = range(len(d))
                ax.errorbar(d["delta"], ys,
                            xerr=[d["delta"] - d["ci_lo"], d["ci_hi"] - d["delta"]],
                            fmt="o", color="crimson", capsize=3)
                ax.set_yticks(list(ys))
                ax.set_yticklabels(d["cell"], fontsize=8)
                ax.axvline(0, color="grey", lw=0.5)
                ax.set_xlabel(f"Δoverlap@10 = neutral − biased  ({pair})")
                ax.set_title(pair)
                ax.grid(alpha=0.25)
            fig.suptitle("Figure E delta - prompt effect on order-stability")
            fig.tight_layout()
            out2 = plots_dir / "figure_e_order_overlap_biased_vs_neutral.png"
            fig.savefig(out2, dpi=200)
            plt.close(fig)
            print(f"[fig E] -> {out2}")
            outs.append(out2)
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
    figure_e_order_sensitivity(out, root)
    print("[make_figures] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
