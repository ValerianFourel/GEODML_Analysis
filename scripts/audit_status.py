#!/usr/bin/env python3
"""Audit pipeline status: per-cell completion, queue state, post-processing.

Walks interpretability/output/ and prints, stage by stage:
  - which (model x treatment x variant) and (model x frame x variant) cells
    finished (`.done_*` marker present), how big the output CSVs are
  - which corresponding SLURM jobs are still queued or running
  - whether post-processing (merged ablation CSVs, figures) exists
  - one overall progress percentage

Run from the repo root:
    python scripts/audit_status.py                    # both variants
    python scripts/audit_status.py --variant biased
    python scripts/audit_status.py --variant neutral

Lives at scripts/audit_status.py so it ships with the pipeline.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "interpretability" / "output"

MODELS = [
    "Llama-3.3-70B-Instruct",
    "Qwen2.5-72B-Instruct",
]
TREATMENTS = [
    "T7_source_earned",
    "T5_topical_comp",
    "T3_structured_data_new",
    "T2a_question_headings",
    "T6_freshness",
    "T1b_stats_density",
]
FRAMES = ["full", "robust_winners"]
VARIANTS_DEFAULT = ["biased", "neutral"]

PLOTS = [
    "figure_a_ablation_full.png",
    "figure_a_ablation_rw.png",
    "figure_b_saliency_full.png",
    "figure_b_saliency_rw.png",
    "figure_c_probing.png",
]


def _ablation_dir(t: str, m: str, variant: str) -> Path:
    return OUT / f"ablation_{t}_{m}_{variant}"


def _saliency_dir(m: str, variant: str) -> Path:
    return OUT / f"saliency_{m}_{variant}"


def _probing_dir(m: str, variant: str) -> Path:
    return OUT / f"probing_{m}_{variant}"


def _weights_dir(m: str, variant: str) -> Path:
    return OUT / f"weights_{m}_{variant}"


# ---------- helpers ----------------------------------------------------------

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
USE_COLOR = sys.stdout.isatty()


def c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if USE_COLOR else text


def rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return max(0, sum(1 for _ in f) - 1)


def size_h(path: Path) -> str:
    if not path.exists():
        return "-"
    n = path.stat().st_size
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}T"


@dataclass
class QueueState:
    by_name: dict[str, list[tuple[str, str]]]  # name -> [(jobid, state)]

    @classmethod
    def collect(cls) -> "QueueState":
        if not shutil.which("squeue"):
            return cls(by_name={})
        try:
            user = os.environ.get("USER") or subprocess.check_output(
                ["whoami"], text=True
            ).strip()
            out = subprocess.check_output(
                ["squeue", "-u", user, "-h", "-o", "%i|%j|%t"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return cls(by_name={})
        by_name: dict[str, list[tuple[str, str]]] = {}
        for line in out.splitlines():
            if not line.strip():
                continue
            jid, name, state = line.split("|", 2)
            by_name.setdefault(name, []).append((jid, state))
        return cls(by_name=by_name)

    def status_for(self, name: str) -> str:
        entries = self.by_name.get(name, [])
        if not entries:
            return ""
        states = [s for _, s in entries]
        if "R" in states:
            return c("RUN", YELLOW)
        if "CG" in states:
            return c("CG ", YELLOW)
        if "PD" in states:
            return c("PD ", DIM)
        return c(",".join(states), DIM)


def cell(done: bool, queued: str, ok: bool = True) -> str:
    if done and ok:
        return c("OK ", GREEN)
    if done and not ok:
        return c("OK?", RED)
    if queued:
        return queued
    return c(" - ", RED)


def header(text: str) -> None:
    bar = "─" * 76
    print(f"\n{c(bar, DIM)}")
    print(c(text, BOLD))
    print(c(bar, DIM))


# ---------- stage audits -----------------------------------------------------

def audit_ablation(q: QueueState, variants: list[str]) -> tuple[int, int]:
    n_cells = len(MODELS) * len(TREATMENTS)
    header(f"Ablation  ({len(MODELS)} models x {len(TREATMENTS)} treatments x "
           f"{len(variants)} variant(s) = {n_cells * len(variants)} jobs)")

    done_n = 0
    total = 0
    for variant in variants:
        print(c(f"  [{variant}]", BOLD))
        print(f"    {'treatment':<24s}", end="")
        for m in MODELS:
            print(f"  {m[:18]:<18s}", end="")
        print()
        for t in TREATMENTS:
            print(f"    {t:<24s}", end="")
            for m in MODELS:
                total += 1
                outdir = _ablation_dir(t, m, variant)
                done = (outdir / f".done_{m}_{t}").exists()
                full = outdir / "ablation_results_full.csv"
                rw = outdir / "ablation_results_rw.csv"
                ok = full.exists() and rw.exists()
                qstr = q.status_for(f"abl-{m}-{t}-{variant}")
                tag = cell(done, qstr, ok=ok)
                if done and ok:
                    done_n += 1
                detail = f"f={rows(full):>5d} r={rows(rw):>5d}"
                print(f"  {tag} {detail:<14s}", end="")
            print()
    return done_n, total


def audit_saliency(q: QueueState, variants: list[str]) -> tuple[int, int]:
    n_cells = len(MODELS) * len(FRAMES)
    header(f"Saliency  ({len(MODELS)} models x {len(FRAMES)} frames x "
           f"{len(variants)} variant(s) = {n_cells * len(variants)} jobs)")
    suffix = {"full": "_full", "robust_winners": "_rw"}
    done_n = 0
    total = 0
    for variant in variants:
        print(c(f"  [{variant}]", BOLD))
        print(f"    {'frame':<18s}", end="")
        for m in MODELS:
            print(f"  {m[:24]:<24s}", end="")
        print()
        for f in FRAMES:
            print(f"    {f:<18s}", end="")
            for m in MODELS:
                total += 1
                outdir = _saliency_dir(m, variant)
                done = (outdir / f".done_{m}_{f}").exists()
                scores = outdir / f"saliency_scores{suffix[f]}.csv"
                summary = outdir / f"saliency_summary{suffix[f]}.csv"
                ok = scores.exists() and summary.exists()
                qstr = q.status_for(f"sal-{m}-{f}-{variant}")
                tag = cell(done, qstr, ok=ok)
                if done and ok:
                    done_n += 1
                detail = f"sc={rows(scores):>6d} sm={rows(summary):>3d}"
                print(f"  {tag} {detail:<20s}", end="")
            print()
    return done_n, total


def audit_probing(q: QueueState, variants: list[str]) -> tuple[int, int]:
    header(f"Probing  ({len(MODELS)} models x {len(variants)} variant(s), frame=both)")
    done_n = 0
    total = 0
    for variant in variants:
        print(c(f"  [{variant}]", BOLD))
        for m in MODELS:
            total += 1
            outdir = _probing_dir(m, variant)
            done = (outdir / f".done_{m}").exists()
            results = outdir / "probing_results.csv"
            ok = results.exists()
            qstr = q.status_for(f"prob-{m}-{variant}")
            tag = cell(done, qstr, ok=ok)
            if done and ok:
                done_n += 1
            n = rows(results)
            # Expected ≈ 32 layers * 4 treatments * 2 pooling * 2 frames = 512
            progress = f"{n:>4d}/512" if n else "    -"
            print(f"    {m:<28s}  {tag}  rows={progress}  ({size_h(results)})")
    return done_n, total


def audit_weights(q: QueueState, variants: list[str]) -> tuple[int, int]:
    header(f"Weight analysis  ({len(MODELS)} models x {len(variants)} variant(s))")
    done_n = 0
    total = 0
    for variant in variants:
        print(c(f"  [{variant}]", BOLD))
        for m in MODELS:
            total += 1
            outdir = _weights_dir(m, variant)
            done = (outdir / f".done_{m}").exists()
            lens = outdir / "logit_lens.csv"
            heads = outdir / "attention_heads.csv"
            ok = lens.exists() and heads.exists()
            qstr = q.status_for(f"wgt-{m}-{variant}")
            tag = cell(done, qstr, ok=ok)
            if done and ok:
                done_n += 1
            print(f"    {m:<28s}  {tag}  lens={rows(lens):>5d}  "
                  f"heads={rows(heads):>6d}  ({size_h(lens)} / {size_h(heads)})")
    return done_n, total


def audit_postprocess() -> tuple[int, int]:
    header("Post-processing (merged ablation CSVs + figures)")
    done_n = 0
    total = 0

    # Merged ablation
    for tag, suf in [("ablation_results_full.csv", "_full"),
                     ("ablation_results_rw.csv", "_rw")]:
        total += 1
        p = OUT / tag
        partitions = sum(
            1 for d in OUT.glob("ablation_*/")
            if (d / f"ablation_results{suf}.csv").exists()
        )
        if p.exists():
            done_n += 1
            mark = c("OK ", GREEN)
        else:
            mark = c(" - ", RED)
        hint = f"  (run scripts/slurm/merge_ablation.sh; {partitions}/12 partitions ready)" \
               if not p.exists() else ""
        print(f"  {mark} {tag:<32s}  rows={rows(p):>5d}  ({size_h(p)}){hint}")

    # Figures
    plots_dir = OUT / "plots"
    for plot in PLOTS:
        total += 1
        p = plots_dir / plot
        if p.exists():
            done_n += 1
            mark = c("OK ", GREEN)
        else:
            mark = c(" - ", RED)
        print(f"  {mark} plots/{plot:<30s}  ({size_h(p)})")
    if not all((plots_dir / p).exists() for p in PLOTS):
        print(f"  {c('hint:', DIM)} python -m interpretability.make_figures")
    return done_n, total


def headline_signs() -> None:
    """If merged ablation CSVs exist, print mean ablation_delta per treatment."""
    header("Headline sanity (per-treatment mean ablation_delta, by frame)")
    try:
        import pandas as pd
    except ImportError:
        print(f"  {c('skip:', DIM)} pandas not installed")
        return

    for tag, label in [("ablation_results_full.csv", "full"),
                       ("ablation_results_rw.csv", "rw")]:
        p = OUT / tag
        if not p.exists():
            print(f"  {c('skip:', DIM)} {tag} not present yet")
            continue
        df = pd.read_csv(p)
        if df.empty:
            print(f"  {c('skip:', DIM)} {tag} has 0 rows")
            continue
        print(f"  [{label}]  n={len(df)}")
        for t in TREATMENTS:
            sub = df[df["treatment"] == t]
            if sub.empty:
                continue
            mean = sub["ablation_delta"].mean()
            expected = (
                "(expect +)" if t == "T7_source_earned"
                else "(expect -)" if t == "T5_topical_comp"
                else ""
            )
            sign_ok = (
                (t == "T7_source_earned" and mean > 0)
                or (t == "T5_topical_comp" and mean < 0)
                or t not in ("T7_source_earned", "T5_topical_comp")
            )
            color = GREEN if sign_ok else RED
            print(f"    {t:<26s} mean={mean:+.3f}  n={len(sub):>5d}  "
                  f"{c(expected, color)}")


# ---------- main -------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("biased", "neutral", "both"),
                    default="both",
                    help="Which prompt variant to audit (default: both).")
    args = ap.parse_args()

    if not OUT.exists():
        print(f"no {OUT} — run from repo root after at least one job has started")
        return 1

    variants = VARIANTS_DEFAULT if args.variant == "both" else [args.variant]

    q = QueueState.collect()
    if not q.by_name and shutil.which("squeue"):
        print(c("(no jobs in your queue)", DIM))

    sections = [
        audit_ablation(q, variants),
        audit_saliency(q, variants),
        audit_probing(q, variants),
        audit_weights(q, variants),
        audit_postprocess(),
    ]
    done = sum(d for d, _ in sections)
    total = sum(t for _, t in sections)
    pct = 100.0 * done / total if total else 0.0

    headline_signs()

    header("Summary")
    bar_w = 40
    filled = int(bar_w * done / total) if total else 0
    bar = "#" * filled + "." * (bar_w - filled)
    color = GREEN if done == total else YELLOW if done else RED
    print(f"  [{c(bar, color)}]  {done}/{total}  ({pct:.0f}%)")

    queued = sum(len(v) for v in q.by_name.values())
    if queued:
        running = sum(
            1 for entries in q.by_name.values()
            for _, st in entries if st == "R"
        )
        pending = sum(
            1 for entries in q.by_name.values()
            for _, st in entries if st == "PD"
        )
        print(f"  queue: {running} R, {pending} PD, {queued} total")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
