#!/usr/bin/env python3
"""Audit pipeline artifacts (Stages A-D) for biased + neutral variants.

Produces a side-by-side report of every artifact the port pipeline writes
under ``$GEODML_DATA_ROOT`` (defaults to ``geodml_data/``):

  Stage A  data/runs/{engine}_{model}_serp{N}_top{K}_{variant}/phase2/keywords.jsonl
  Stage B  data/features/features_{engine}_top{pool}.parquet         (variant-agnostic)
  Stage C  data/main/full_experiment_data_{variant}.parquet
  Stage D  data/dml_results/dml_results_long_{variant}.parquet

Designed to be run before AND after a pipeline run so you can confirm what
actually changed:

    # before kicking off the neutral chain
    python scripts/audit_pipeline.py --save audits/before_neutral.json

    # ... wait for jobs ...

    # after everything finishes
    python scripts/audit_pipeline.py --save audits/after_neutral.json

    # diff the two
    python scripts/audit_pipeline.py --compare audits/before_neutral.json audits/after_neutral.json

All paths are derived from ``interpretability.pipeline.config`` so the script
stays in sync with the pipeline modules. No HTTP, no SLURM job submission.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from interpretability.pipeline import config as C  # noqa: E402

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN = "\033[32m"; YELLOW = "\033[33m"; RED = "\033[31m"
DIM = "\033[2m"; BOLD = "\033[1m"; CYAN = "\033[36m"; RESET = "\033[0m"
USE_COLOR = sys.stdout.isatty()


def col(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if USE_COLOR else text


def header(text: str) -> None:
    bar = "─" * 78
    print(f"\n{col(bar, DIM)}")
    print(col(text, BOLD))
    print(col(bar, DIM))


def size_h(n: int | None) -> str:
    if not n:
        return "-"
    x = float(n)
    for unit in ("B", "K", "M", "G"):
        if x < 1024:
            return f"{x:.0f}{unit}"
        x /= 1024
    return f"{x:.0f}T"


def jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def parquet_meta(path: Path) -> dict[str, Any]:
    """Return {rows, n_cols, size}; safe on missing/corrupt files."""
    if not path.exists():
        return {"rows": 0, "n_cols": 0, "size": 0}
    size = path.stat().st_size
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        return {"rows": pf.metadata.num_rows, "n_cols": pf.schema.num_columns, "size": size}
    except Exception:
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            return {"rows": len(df), "n_cols": df.shape[1], "size": size}
        except Exception as e:
            return {"rows": -1, "n_cols": -1, "size": size, "error": str(e)}


# ── Snapshot dataclasses (JSON-serialisable) ─────────────────────────────────

@dataclass
class InputRow:
    kind: str          # "serp" | "html_tar"
    engine: str
    pool: int
    path: str
    rows: int
    size: int


@dataclass
class StageARow:
    model: str
    engine: str
    pool: int
    variant: str
    run_id: str
    jsonl_path: str
    keywords: int
    rankings_csv_rows: int
    ckpt_present: bool
    run_dir_size: int


@dataclass
class StageBRow:
    engine: str
    pool: int
    path: str
    rows: int
    n_cols: int
    size: int


@dataclass
class StageCInfo:
    variant: str
    path: str
    rows: int
    n_cols: int
    size: int


@dataclass
class StageDInfo:
    variant: str
    path: str
    rows: int
    n_cols: int
    size: int
    treatments: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    learners: list[str] = field(default_factory=list)
    subsets: list[str] = field(default_factory=list)
    pooled_plr_lgbm_rank_delta: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class OrderProbeRow:
    model: str
    engine: str
    pool: int
    variant: str
    seed: int
    run_id: str
    jsonl_path: str
    keywords: int
    size: int


@dataclass
class InterpRow:
    """One per (stage, model, variant[, treatment|frame]) cell."""
    stage: str          # ablation | saliency | probing | weights
    model: str
    variant: str
    treatment: str      # only for ablation; "" otherwise
    frame: str          # "full" / "robust_winners" for saliency/probing; "" otherwise
    out_dir: str
    done_marker: bool
    main_csv_rows: int  # primary output rows (ablation_results_full / saliency_summary / probing_results / logit_lens)
    main_csv_size: int
    rw_csv_rows: int    # only meaningful for ablation/saliency, else 0


# Treatments and frames are kept here as a small constant set; matches what
# audit_status.py and dispatch_all.sh already iterate over.
INTERP_TREATMENTS = [
    "T7_source_earned", "T5_topical_comp", "T3_structured_data_new",
    "T2a_question_headings", "T6_freshness", "T1b_stats_density",
]
INTERP_FRAMES = ["full", "robust_winners"]
_FRAME_SUFFIX = {"full": "_full", "robust_winners": "_rw"}


@dataclass
class Snapshot:
    timestamp: str
    data_root: str
    git_commit: str | None
    git_branch: str | None
    queue_summary: dict[str, int]
    inputs: list[InputRow]
    stage_a: list[StageARow]
    stage_b: list[StageBRow]
    stage_c: list[StageCInfo]
    stage_d: list[StageDInfo]
    order_probe: list[OrderProbeRow]
    order_probe_summary: dict[str, Any]
    interp: list[InterpRow]


# ── Collectors ───────────────────────────────────────────────────────────────

def _git(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _queue_summary() -> dict[str, int]:
    if not shutil.which("squeue"):
        return {}
    user = os.environ.get("USER") or "unknown"
    try:
        out = subprocess.check_output(
            ["squeue", "-u", user, "-h", "-o", "%t"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return {}
    summary: dict[str, int] = {}
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        summary[s] = summary.get(s, 0) + 1
    return summary


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def collect_inputs(data_root: Path) -> list[InputRow]:
    """Pipeline INPUTS — cached SERP parquets + HTML tarballs. If these are
    missing, every Stage A job silently produces zero keywords.
    """
    rows: list[InputRow] = []
    for engine in C.ENGINES:
        for serp_n, _ in C.POOL_SIZES:
            p = C.serp_path(engine, serp_n, data_root)
            m = parquet_meta(p) if p.exists() else {"rows": 0, "size": 0}
            rows.append(InputRow(
                kind="serp", engine=engine, pool=serp_n,
                path=str(p), rows=m["rows"], size=m["size"],
            ))
    runs_root = data_root / "data" / "runs"
    seen: set[tuple[str, int]] = set()
    if runs_root.exists():
        for d in runs_root.glob(f"*_serp*"):
            tar = d / "phase2" / "html_cache.tar.gz"
            if not tar.exists():
                continue
            name = d.name
            for engine in C.ENGINES:
                if name.startswith(f"{engine}_"):
                    for serp_n, _ in C.POOL_SIZES:
                        if f"_serp{serp_n}_" in name and (engine, serp_n) not in seen:
                            seen.add((engine, serp_n))
                            rows.append(InputRow(
                                kind="html_tar", engine=engine, pool=serp_n,
                                path=str(tar), rows=0,
                                size=tar.stat().st_size,
                            ))
                            break
                    break
    return rows


def collect_stage_a(data_root: Path, variants: list[str]) -> list[StageARow]:
    rows: list[StageARow] = []
    for model_id in C.LLM_MODELS:
        model_short = C.short_model_name(model_id)
        for engine in C.ENGINES:
            for serp_n, top_n in C.POOL_SIZES:
                for variant in variants:
                    run_id = C.run_label_with_variant(engine, model_id, serp_n, top_n, variant)
                    run_dir = data_root / "data" / "runs" / run_id
                    phase2 = run_dir / "phase2"
                    jsonl = phase2 / "keywords.jsonl"
                    rankings = phase2 / "rankings.csv"
                    ckpt = phase2 / ".rerank_ckpt.json"
                    rows.append(StageARow(
                        model=model_short,
                        engine=engine,
                        pool=serp_n,
                        variant=variant,
                        run_id=run_id,
                        jsonl_path=str(jsonl),
                        keywords=jsonl_rows(jsonl),
                        rankings_csv_rows=max(0, jsonl_rows(rankings) - 1) if rankings.exists() else 0,
                        ckpt_present=ckpt.exists(),
                        run_dir_size=_dir_size(run_dir),
                    ))
    return rows


def collect_stage_b(data_root: Path) -> list[StageBRow]:
    rows: list[StageBRow] = []
    feat_dir = data_root / "data" / "features"
    for engine in C.ENGINES:
        for serp_n, _ in C.POOL_SIZES:
            p = feat_dir / f"features_{engine}_top{serp_n}.parquet"
            m = parquet_meta(p)
            rows.append(StageBRow(
                engine=engine, pool=serp_n,
                path=str(p),
                rows=m["rows"], n_cols=m["n_cols"], size=m["size"],
            ))
    return rows


def collect_stage_c(data_root: Path, variants: list[str]) -> list[StageCInfo]:
    out: list[StageCInfo] = []
    for v in variants:
        p = C.main_table_path(v, data_root)
        m = parquet_meta(p)
        out.append(StageCInfo(
            variant=v, path=str(p),
            rows=m["rows"], n_cols=m["n_cols"], size=m["size"],
        ))
    return out


def _summarise_dml(path: Path, m: dict[str, Any]) -> dict[str, Any]:
    """Read the dml parquet (lazily) and pull a small headline subset."""
    extra: dict[str, Any] = {
        "treatments": [], "methods": [], "learners": [], "subsets": [],
        "pooled_plr_lgbm_rank_delta": [],
    }
    if m["rows"] <= 0:
        return extra
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        for k in ("treatment", "method", "learner", "subset"):
            if k in df.columns:
                extra[f"{k}s"] = sorted(df[k].dropna().unique().tolist())
        sub = df.copy()
        for col_name, val in [("subset", "POOLED"), ("method", "plr"),
                              ("learner", "lgbm"), ("outcome", "rank_delta")]:
            if col_name in sub.columns:
                sub = sub[sub[col_name] == val]
        keep_cols = [c for c in ("treatment", "coef", "se", "p_val", "sig_stars")
                     if c in sub.columns]
        if keep_cols:
            extra["pooled_plr_lgbm_rank_delta"] = (
                sub[keep_cols].sort_values("treatment").to_dict("records")
            )
    except Exception as e:
        extra["error"] = str(e)
    return extra


def collect_stage_d(data_root: Path, variants: list[str]) -> list[StageDInfo]:
    out: list[StageDInfo] = []
    for v in variants:
        p = C.dml_results_path(v, data_root)
        m = parquet_meta(p)
        extra = _summarise_dml(p, m)
        out.append(StageDInfo(
            variant=v, path=str(p),
            rows=m["rows"], n_cols=m["n_cols"], size=m["size"],
            treatments=extra.get("treatments", []),
            methods=extra.get("methods", []),
            learners=extra.get("learners", []),
            subsets=extra.get("subsets", []),
            pooled_plr_lgbm_rank_delta=extra.get("pooled_plr_lgbm_rank_delta", []),
        ))
    return out


def collect_order_probe(data_root: Path, variants: list[str],
                        seeds: list[int]) -> list[OrderProbeRow]:
    rows: list[OrderProbeRow] = []
    for variant in variants:
        for model_id in C.LLM_MODELS:
            model_short = C.short_model_name(model_id)
            for engine in C.ENGINES:
                for serp_n, top_n in C.POOL_SIZES:
                    run_id = C.run_label_with_variant(
                        engine, model_id, serp_n, top_n, variant,
                    )
                    for seed in seeds:
                        p = data_root / "data" / "order_probe" / f"{run_id}_seed{seed}.jsonl"
                        rows.append(OrderProbeRow(
                            model=model_short,
                            engine=engine,
                            pool=serp_n,
                            variant=variant,
                            seed=seed,
                            run_id=run_id,
                            jsonl_path=str(p),
                            keywords=jsonl_rows(p),
                            size=p.stat().st_size if p.exists() else 0,
                        ))
    return rows


def collect_order_probe_summary(data_root: Path) -> dict[str, Any]:
    """Tiny headline read of order_probe_summary.parquet if present."""
    p = data_root / "data" / "order_probe" / "order_probe_summary.parquet"
    if not p.exists():
        return {"path": str(p), "exists": False, "rows": 0}
    m = parquet_meta(p)
    out: dict[str, Any] = {
        "path": str(p), "exists": True,
        "rows": m["rows"], "size": m["size"],
        "headline": [],
    }
    if m["rows"] <= 0:
        return out
    try:
        import pandas as pd
        df = pd.read_parquet(p)
        if "K" in df.columns:
            sub = df[df.K == 10]
            head = (sub.groupby(["variant", "ordering_pair"])
                       .agg(mean_jacc=("jaccard", "mean"),
                            mean_oak=("overlap_at_k", "mean"),
                            n=("keyword", "count"))
                       .reset_index()
                       .round(3))
            out["headline"] = head.to_dict("records")
    except Exception as e:
        out["error"] = str(e)
    return out


def collect_interp(variants: list[str]) -> list[InterpRow]:
    """Walk interpretability/output/ for variant-suffixed dirs (Stage F)."""
    out_root = REPO_ROOT / "interpretability" / "output"
    rows: list[InterpRow] = []
    for variant in variants:
        for model_id in C.LLM_MODELS:
            m = C.short_model_name(model_id)
            # ablation: per-treatment dir
            for t in INTERP_TREATMENTS:
                d = out_root / f"ablation_{t}_{m}_{variant}"
                full = d / "ablation_results_full.csv"
                rw = d / "ablation_results_rw.csv"
                rows.append(InterpRow(
                    stage="ablation", model=m, variant=variant,
                    treatment=t, frame="",
                    out_dir=str(d),
                    done_marker=(d / f".done_{m}_{t}").exists(),
                    main_csv_rows=jsonl_rows(full),
                    main_csv_size=full.stat().st_size if full.exists() else 0,
                    rw_csv_rows=jsonl_rows(rw),
                ))
            # saliency: per-frame, but stored in a single per-model dir
            for f in INTERP_FRAMES:
                d = out_root / f"saliency_{m}_{variant}"
                summary = d / f"saliency_summary{_FRAME_SUFFIX[f]}.csv"
                rows.append(InterpRow(
                    stage="saliency", model=m, variant=variant,
                    treatment="", frame=f,
                    out_dir=str(d),
                    done_marker=(d / f".done_{m}_{f}").exists(),
                    main_csv_rows=jsonl_rows(summary),
                    main_csv_size=summary.stat().st_size if summary.exists() else 0,
                    rw_csv_rows=0,
                ))
            # probing: one CSV per model (frame=both is in the same file)
            d = out_root / f"probing_{m}_{variant}"
            results = d / "probing_results.csv"
            rows.append(InterpRow(
                stage="probing", model=m, variant=variant,
                treatment="", frame="both",
                out_dir=str(d),
                done_marker=(d / f".done_{m}").exists(),
                main_csv_rows=jsonl_rows(results),
                main_csv_size=results.stat().st_size if results.exists() else 0,
                rw_csv_rows=0,
            ))
            # weights: logit_lens.csv as primary
            d = out_root / f"weights_{m}_{variant}"
            lens = d / "logit_lens.csv"
            rows.append(InterpRow(
                stage="weights", model=m, variant=variant,
                treatment="", frame="",
                out_dir=str(d),
                done_marker=(d / f".done_{m}").exists(),
                main_csv_rows=jsonl_rows(lens),
                main_csv_size=lens.stat().st_size if lens.exists() else 0,
                rw_csv_rows=0,
            ))
    return rows


def collect_snapshot(data_root: Path, variants: list[str],
                     seeds: list[int] | None = None) -> Snapshot:
    seeds = seeds if seeds is not None else [42, 123]
    return Snapshot(
        timestamp=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        data_root=str(data_root),
        git_commit=_git(["git", "rev-parse", "--short", "HEAD"]),
        git_branch=_git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        queue_summary=_queue_summary(),
        inputs=collect_inputs(data_root),
        stage_a=collect_stage_a(data_root, variants),
        stage_b=collect_stage_b(data_root),
        stage_c=collect_stage_c(data_root, variants),
        stage_d=collect_stage_d(data_root, variants),
        order_probe=collect_order_probe(data_root, variants, seeds),
        order_probe_summary=collect_order_probe_summary(data_root),
        interp=collect_interp(variants),
    )


# ── Pretty-printers ──────────────────────────────────────────────────────────

def _ok(value: int | bool, *, expected_min: int | None = None) -> str:
    if isinstance(value, bool):
        return col("OK ", GREEN) if value else col(" - ", DIM)
    if value < 0:
        return col("ERR", RED)
    if value == 0:
        return col(" - ", DIM)
    if expected_min is not None and value < expected_min:
        return col("LOW", YELLOW)
    return col("OK ", GREEN)


def print_inputs(rows: list[InputRow]) -> None:
    n_serp = len(C.ENGINES) * len(C.POOL_SIZES)
    header(f"Inputs — cached SERP parquets ({n_serp} expected) + HTML tarballs")
    for r in rows:
        if r.kind == "serp":
            tag = _ok(r.rows, expected_min=10)
            print(f"  {tag}  serp     {r.engine:<8s}  pool={r.pool:<3d}  "
                  f"rows={r.rows:>6d}  ({size_h(r.size)})  "
                  f"{col(Path(r.path).name, DIM)}")
    html_rows = [r for r in rows if r.kind == "html_tar"]
    if html_rows:
        print(f"  {col('html_cache.tar.gz tarballs found:', DIM)}")
        for r in html_rows:
            tag = col("OK ", GREEN)
            print(f"  {tag}  html     {r.engine:<8s}  pool={r.pool:<3d}  "
                  f"({size_h(r.size)})  {col(Path(r.path).parent.parent.name, DIM)}")
    serp_missing = [r for r in rows if r.kind == "serp" and r.rows == 0]
    if serp_missing:
        print(col(f"  WARNING: {len(serp_missing)} SERP parquet(s) missing — "
                  f"rerank will produce 0 keywords for those cells", RED))


def print_stage_a(rows: list[StageARow], variants: list[str]) -> None:
    n_cells = len(C.LLM_MODELS) * len(C.ENGINES) * len(C.POOL_SIZES)
    header(f"Stage A — rerank   ({n_cells} cells × {len(variants)} variant(s) "
           f"= {n_cells * len(variants)} jsonl files)")
    print(f"  {'model':<22s}  {'engine':<8s}  {'pool':<5s}  ", end="")
    for v in variants:
        print(f"  {v:<28s}", end="")
    print()
    print(f"  {'-'*22}  {'-'*8}  {'-'*5}  ", end="")
    for _ in variants:
        print(f"  {'-'*28}", end="")
    print()

    by_key: dict[tuple[str, str, int, str], StageARow] = {
        (r.model, r.engine, r.pool, r.variant): r for r in rows
    }
    for model_id in C.LLM_MODELS:
        m = C.short_model_name(model_id)
        for engine in C.ENGINES:
            for serp_n, _ in C.POOL_SIZES:
                print(f"  {m[:22]:<22s}  {engine:<8s}  {serp_n:<5d}  ", end="")
                for v in variants:
                    r = by_key.get((m, engine, serp_n, v))
                    if r is None:
                        print(f"  {col(' - ', DIM):<28s}", end="")
                        continue
                    tag = _ok(r.keywords, expected_min=50)
                    extra = "ckpt" if r.ckpt_present else "    "
                    s = f"{tag} {r.keywords:>4d}kw {extra} ({size_h(r.run_dir_size)})"
                    print(f"  {s:<28s}", end="")
                print()
    # totals
    for v in variants:
        sub = [r for r in rows if r.variant == v]
        n_done = sum(1 for r in sub if r.keywords > 0)
        kw_total = sum(r.keywords for r in sub)
        print(f"  [{v}] cells with keywords: {n_done}/{n_cells}  total keyword-records: {kw_total}")


def print_stage_b(rows: list[StageBRow]) -> None:
    n_expected = len(C.ENGINES) * len(C.POOL_SIZES)
    header(f"Stage B — features (variant-agnostic; {n_expected} parquets expected)")
    for r in rows:
        tag = _ok(r.rows, expected_min=100)
        print(f"  {tag}  {r.engine:<8s}  pool={r.pool:<3d}  "
              f"rows={r.rows:>7d}  cols={r.n_cols:>3d}  ({size_h(r.size)})")
    n_done = sum(1 for r in rows if r.rows > 0)
    print(f"  total: {n_done}/{n_expected} parquets present")


def print_stage_c(rows: list[StageCInfo]) -> None:
    header("Stage C — merged main table (full_experiment_data_{variant}.parquet)")
    for r in rows:
        tag = _ok(r.rows, expected_min=100)
        path = Path(r.path).name
        print(f"  {tag}  {path:<40s}  rows={r.rows:>7d}  cols={r.n_cols:>3d}  ({size_h(r.size)})")


def print_stage_d(rows: list[StageDInfo]) -> None:
    header("Stage D — DML results (dml_results_long_{variant}.parquet)")
    for r in rows:
        tag = _ok(r.rows, expected_min=10)
        path = Path(r.path).name
        print(f"  {tag}  {path:<40s}  rows={r.rows:>5d}  cols={r.n_cols:>3d}  "
              f"({size_h(r.size)})")
        if r.rows > 0:
            print(f"        treatments={len(r.treatments):<2d}  methods={r.methods}  "
                  f"learners={r.learners}  subsets={len(r.subsets)}")


def print_interp(rows: list[InterpRow], variants: list[str]) -> None:
    """Stage F — ablation, saliency, probing, weights, per variant."""
    n_per_var = (
        len(C.LLM_MODELS) * len(INTERP_TREATMENTS)        # ablation
        + len(C.LLM_MODELS) * len(INTERP_FRAMES)          # saliency
        + len(C.LLM_MODELS)                               # probing
        + len(C.LLM_MODELS)                               # weights
    )
    header(f"Stage F — interpretability   "
           f"({n_per_var} cells × {len(variants)} variant(s) "
           f"= {n_per_var * len(variants)} jobs)")

    by_stage: dict[str, list[InterpRow]] = {}
    for r in rows:
        by_stage.setdefault(r.stage, []).append(r)

    for stage in ("ablation", "saliency", "probing", "weights"):
        sr = by_stage.get(stage, [])
        n_done = sum(1 for r in sr if r.main_csv_rows > 0)
        n_total = len(sr)
        n_done_marker = sum(1 for r in sr if r.done_marker)
        print(f"  {stage:<10s}  cells with CSVs={n_done:>3d}/{n_total:<3d}   "
              f"done-markers={n_done_marker:>3d}")
        for variant in variants:
            sub = [r for r in sr if r.variant == variant]
            if not sub:
                continue
            done_v = sum(1 for r in sub if r.main_csv_rows > 0)
            marker_v = sum(1 for r in sub if r.done_marker)
            tag = (col("OK ", GREEN) if done_v == len(sub)
                   else col("PRT", YELLOW) if done_v > 0
                   else col(" - ", DIM))
            # Include the headline row count for the largest cell
            top_rows = max((r.main_csv_rows for r in sub), default=0)
            print(f"             [{variant:<8s}] {tag} cells={done_v:>2d}/{len(sub):<2d} "
                  f"  markers={marker_v:>2d}/{len(sub):<2d}   max_rows={top_rows}")


def print_order_probe(rows: list[OrderProbeRow], variants: list[str],
                       seeds: list[int]) -> None:
    n_cells = len(C.LLM_MODELS) * len(C.ENGINES) * len(C.POOL_SIZES)
    total = n_cells * len(variants) * len(seeds)
    header(f"Stage A' — order probe   ({n_cells} cells × {len(variants)} variant(s) × "
           f"{len(seeds)} seeds = {total} jsonl files)")
    print(f"  {'model':<22s}  {'engine':<8s}  {'pool':<5s}  {'variant':<8s}  ", end="")
    for s in seeds:
        print(f"  seed={s:<5d}", end="")
    print()
    print(f"  {'-'*22}  {'-'*8}  {'-'*5}  {'-'*8}  ", end="")
    for _ in seeds:
        print(f"  {'-'*10}", end="")
    print()

    by_key: dict[tuple[str, str, int, str, int], OrderProbeRow] = {
        (r.model, r.engine, r.pool, r.variant, r.seed): r for r in rows
    }
    for variant in variants:
        for model_id in C.LLM_MODELS:
            m = C.short_model_name(model_id)
            for engine in C.ENGINES:
                for serp_n, _ in C.POOL_SIZES:
                    print(f"  {m[:22]:<22s}  {engine:<8s}  {serp_n:<5d}  "
                          f"{variant:<8s}  ", end="")
                    for s in seeds:
                        r = by_key.get((m, engine, serp_n, variant, s))
                        if r is None or r.keywords == 0:
                            print(f"  {col(' - ', DIM):<10s}", end="")
                        else:
                            tag = _ok(r.keywords, expected_min=10)
                            print(f"  {tag} {r.keywords:>5d}", end="")
                    print()
    n_done = sum(1 for r in rows if r.keywords > 0)
    kw_total = sum(r.keywords for r in rows)
    print(f"  cells with output: {n_done}/{total}  total keyword-records: {kw_total}")


def print_order_probe_summary(summary: dict[str, Any]) -> None:
    header("Stage A' — order probe analysis (order_probe_summary.parquet)")
    if not summary.get("exists"):
        print(col("  not yet generated. After all 32 jobs finish, run:", DIM))
        print(col("    python -m interpretability.pipeline.order_probe_analyze", DIM))
        return
    rows = summary.get("rows", 0)
    if rows <= 0:
        print(col(f"  empty parquet at {Path(summary['path']).name} — analyze produced no rows.", YELLOW))
        return
    print(f"  {col('rows:', DIM)} {rows}    "
          f"{col('K=10 headline (mean overlap by variant × ordering_pair):', DIM)}")
    head = summary.get("headline", [])
    if not head:
        print(col("  (headline missing — parquet probably has unexpected columns)", YELLOW))
        return
    print(f"  {'variant':<8s}  {'ordering_pair':<28s}  {'mean_jacc':>9s}  {'mean_oak':>9s}  {'n':>6s}")
    for r in sorted(head, key=lambda x: (x["variant"], x["ordering_pair"])):
        print(f"  {r['variant']:<8s}  {r['ordering_pair']:<28s}  "
              f"{r['mean_jacc']:>9.3f}  {r['mean_oak']:>9.3f}  {r['n']:>6d}")


def print_headline(stage_d: list[StageDInfo]) -> None:
    """Cross-variant headline: POOLED+plr+lgbm+rank_delta coefficients."""
    header("Headline — POOLED, plr, lgbm, rank_delta  (Δ = neutral − biased)")
    by_var = {r.variant: r for r in stage_d}
    biased = by_var.get("biased")
    neutral = by_var.get("neutral")

    if not biased and not neutral:
        print("  (no DML output for either variant — run Stage D first)")
        return

    def to_map(r: StageDInfo | None) -> dict[str, dict[str, Any]]:
        if r is None or not r.pooled_plr_lgbm_rank_delta:
            return {}
        return {row["treatment"]: row for row in r.pooled_plr_lgbm_rank_delta}

    mb, mn = to_map(biased), to_map(neutral)
    treatments = sorted(set(mb) | set(mn))
    if not treatments:
        print("  (no POOLED+plr+lgbm+rank_delta rows in either variant)")
        return

    print(f"  {'treatment':<28s}  {'biased':<22s}  {'neutral':<22s}  {'Δ':<10s}")
    print(f"  {'-'*28}  {'-'*22}  {'-'*22}  {'-'*10}")
    for t in treatments:
        b = mb.get(t)
        n = mn.get(t)

        def fmt(row: dict[str, Any] | None) -> str:
            if row is None:
                return "-"
            stars = row.get("sig_stars") or ""
            return f"{row['coef']:+.3f}{stars:<3s} (se={row.get('se', 0):.3f})"

        b_s = fmt(b)
        n_s = fmt(n)
        if b is not None and n is not None:
            delta = n["coef"] - b["coef"]
            d_s = f"{delta:+.3f}"
        else:
            d_s = "-"
        print(f"  {t:<28s}  {b_s:<22s}  {n_s:<22s}  {d_s:<10s}")


def print_meta(snap: Snapshot) -> None:
    header("Audit context")
    print(f"  timestamp:   {snap.timestamp}")
    print(f"  data_root:   {snap.data_root}")
    print(f"  git:         {snap.git_branch} @ {snap.git_commit}")
    if snap.queue_summary:
        q = "  ".join(f"{k}={v}" for k, v in sorted(snap.queue_summary.items()))
        print(f"  squeue:      {q}")
    else:
        print(f"  squeue:      {col('(empty / unavailable)', DIM)}")


# ── Compare mode ─────────────────────────────────────────────────────────────

def _diff_count(before: int, after: int) -> str:
    if before == after:
        return col(f"={after}", DIM)
    delta = after - before
    sign = "+" if delta > 0 else ""
    color = GREEN if delta > 0 else RED
    return col(f"{before}→{after} ({sign}{delta})", color)


def compare_snapshots(before: dict, after: dict) -> None:
    header("Snapshot diff")
    print(f"  before: {before['timestamp']}  ({before.get('git_commit')})")
    print(f"  after:  {after['timestamp']}  ({after.get('git_commit')})")

    # Stage A
    header("Stage A — rerank changes")
    bmap = {(r["model"], r["engine"], r["pool"], r["variant"]): r for r in before["stage_a"]}
    amap = {(r["model"], r["engine"], r["pool"], r["variant"]): r for r in after["stage_a"]}
    keys = sorted(set(bmap) | set(amap))
    any_change = False
    for k in keys:
        b = bmap.get(k, {"keywords": 0})
        a = amap.get(k, {"keywords": 0})
        if b["keywords"] != a["keywords"]:
            any_change = True
            print(f"  {k[3]:<8s}  {k[0][:22]:<22s} {k[1]:<8s} pool={k[2]:<3d}  "
                  f"keywords {_diff_count(b['keywords'], a['keywords'])}")
    if not any_change:
        print(col("  no change", DIM))

    # Stage B
    header("Stage B — features changes")
    bmap2 = {(r["engine"], r["pool"]): r for r in before["stage_b"]}
    amap2 = {(r["engine"], r["pool"]): r for r in after["stage_b"]}
    any_change = False
    for k in sorted(set(bmap2) | set(amap2)):
        b = bmap2.get(k, {"rows": 0})
        a = amap2.get(k, {"rows": 0})
        if b["rows"] != a["rows"]:
            any_change = True
            print(f"  {k[0]:<8s} pool={k[1]:<3d}  rows {_diff_count(b['rows'], a['rows'])}")
    if not any_change:
        print(col("  no change", DIM))

    # Stage C / D
    for stage_key, label in [("stage_c", "Stage C — merged main"),
                             ("stage_d", "Stage D — DML results")]:
        header(f"{label} changes")
        bmap3 = {r["variant"]: r for r in before[stage_key]}
        amap3 = {r["variant"]: r for r in after[stage_key]}
        any_change = False
        for v in sorted(set(bmap3) | set(amap3)):
            b = bmap3.get(v, {"rows": 0})
            a = amap3.get(v, {"rows": 0})
            if b["rows"] != a["rows"]:
                any_change = True
                print(f"  variant={v:<8s}  rows {_diff_count(b['rows'], a['rows'])}")
        if not any_change:
            print(col("  no change", DIM))

    # Stage F — interp
    header("Stage F — interpretability changes")
    def _ikey(r: dict) -> tuple:
        return (r["stage"], r["model"], r["variant"], r.get("treatment", ""),
                r.get("frame", ""))
    bmapF = {_ikey(r): r for r in before.get("interp", [])}
    amapF = {_ikey(r): r for r in after.get("interp", [])}
    any_change = False
    for k in sorted(set(bmapF) | set(amapF)):
        b = bmapF.get(k, {"main_csv_rows": 0, "done_marker": False})
        a = amapF.get(k, {"main_csv_rows": 0, "done_marker": False})
        if b["main_csv_rows"] != a["main_csv_rows"] or b["done_marker"] != a["done_marker"]:
            any_change = True
            stage, model, variant, t, f = k
            tag = f"{stage}/{model[:18]}/{variant}"
            if t:
                tag += f"/{t}"
            elif f and stage != "probing":
                tag += f"/{f}"
            marker = ""
            if b["done_marker"] != a["done_marker"]:
                marker = "  done={}→{}".format(b["done_marker"], a["done_marker"])
            print(f"  {tag:<60s}  rows {_diff_count(b['main_csv_rows'], a['main_csv_rows'])}{marker}")
    if not any_change:
        print(col("  no change", DIM))

    # Order probe
    header("Stage A' — order probe changes")
    bmap4 = {(r["model"], r["engine"], r["pool"], r["variant"], r["seed"]): r
             for r in before.get("order_probe", [])}
    amap4 = {(r["model"], r["engine"], r["pool"], r["variant"], r["seed"]): r
             for r in after.get("order_probe", [])}
    any_change = False
    for k in sorted(set(bmap4) | set(amap4)):
        b = bmap4.get(k, {"keywords": 0})
        a = amap4.get(k, {"keywords": 0})
        if b["keywords"] != a["keywords"]:
            any_change = True
            print(f"  {k[3]:<8s}  {k[0][:22]:<22s} {k[1]:<8s} pool={k[2]:<3d} "
                  f"seed={k[4]:<4d}  keywords {_diff_count(b['keywords'], a['keywords'])}")
    if not any_change:
        print(col("  no change", DIM))

    # Headline coefficient diffs (Stage D only)
    header("Stage D — POOLED+plr+lgbm+rank_delta coefficient changes")
    def hmap(snap: dict) -> dict[tuple[str, str], dict[str, Any]]:
        out: dict[tuple[str, str], dict[str, Any]] = {}
        for r in snap["stage_d"]:
            for row in r.get("pooled_plr_lgbm_rank_delta", []):
                out[(r["variant"], row["treatment"])] = row
        return out
    bh, ah = hmap(before), hmap(after)
    keys = sorted(set(bh) | set(ah))
    any_change = False
    for v, t in keys:
        b = bh.get((v, t))
        a = ah.get((v, t))
        if b is None and a is not None:
            any_change = True
            print(f"  {v:<8s}  {t:<28s}  new: {a['coef']:+.3f}{a.get('sig_stars') or ''}")
        elif a is None and b is not None:
            any_change = True
            print(f"  {v:<8s}  {t:<28s}  removed: was {b['coef']:+.3f}")
        elif b is not None and a is not None:
            if abs(b["coef"] - a["coef"]) > 1e-6:
                any_change = True
                d = a["coef"] - b["coef"]
                print(f"  {v:<8s}  {t:<28s}  {b['coef']:+.3f} → {a['coef']:+.3f}  (Δ={d:+.3f})")
    if not any_change:
        print(col("  no change", DIM))


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant",
                    choices=("biased", "neutral",
                             "biased_passage", "neutral_passage",
                             "biased_rag", "neutral_rag",
                             "both", "all"),
                    default="all",
                    help="Which prompt variant to audit. 'both' = biased+neutral "
                         "(snippet-only); 'all' = all six variants (snippet, "
                         "passage-augmented, RAG-augmented).")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123],
                    help="Order-probe seeds to audit (default: 42 123).")
    ap.add_argument("--data-root", default=None,
                    help="Override $GEODML_DATA_ROOT.")
    ap.add_argument("--save", default=None, metavar="FILE.json",
                    help="Write a JSON snapshot of the audit to FILE.")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE.json", "AFTER.json"),
                    help="Diff two previously-saved snapshots and exit.")
    ap.add_argument("--json", action="store_true",
                    help="Print snapshot as JSON to stdout (no pretty report).")
    args = ap.parse_args()

    if args.compare:
        with open(args.compare[0]) as f:
            before = json.load(f)
        with open(args.compare[1]) as f:
            after = json.load(f)
        compare_snapshots(before, after)
        return 0

    data_root = Path(args.data_root) if args.data_root else C.DEFAULT_DATA_ROOT
    if args.variant == "all":
        variants = [
            "biased", "neutral",
            "biased_passage", "neutral_passage",
            "biased_rag", "neutral_rag",
        ]
    elif args.variant == "both":
        variants = ["biased", "neutral"]
    else:
        variants = [args.variant]

    snap = collect_snapshot(data_root, variants, seeds=args.seeds)

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(asdict(snap), f, indent=2, default=str)
        print(f"[audit] snapshot saved -> {out_path}")

    if args.json:
        json.dump(asdict(snap), sys.stdout, indent=2, default=str)
        print()
        return 0

    print_meta(snap)
    print_inputs(snap.inputs)
    print_stage_a(snap.stage_a, variants)
    print_stage_b(snap.stage_b)
    print_stage_c(snap.stage_c)
    print_stage_d(snap.stage_d)
    print_headline(snap.stage_d)
    seeds_used = sorted({r.seed for r in snap.order_probe})
    if seeds_used:
        print_order_probe(snap.order_probe, variants, seeds_used)
        print_order_probe_summary(snap.order_probe_summary)
    print_interp(snap.interp, variants)

    # Final summary line
    sa_done = sum(1 for r in snap.stage_a if r.keywords > 0)
    sa_total = len(snap.stage_a)
    sb_done = sum(1 for r in snap.stage_b if r.rows > 0)
    sb_total = len(snap.stage_b)
    sc_done = sum(1 for r in snap.stage_c if r.rows > 0)
    sc_total = len(snap.stage_c)
    sd_done = sum(1 for r in snap.stage_d if r.rows > 0)
    sd_total = len(snap.stage_d)
    op_done = sum(1 for r in snap.order_probe if r.keywords > 0)
    op_total = len(snap.order_probe)
    f_done = sum(1 for r in snap.interp if r.main_csv_rows > 0)
    f_total = len(snap.interp)

    header("Summary")
    print(f"  Stage A   {sa_done:>2d}/{sa_total:<2d}   "
          f"Stage B  {sb_done:>2d}/{sb_total:<2d}   "
          f"Stage C  {sc_done:>2d}/{sc_total:<2d}   "
          f"Stage D  {sd_done:>2d}/{sd_total:<2d}   "
          f"Stage F  {f_done:>2d}/{f_total:<2d}   "
          f"Order probe  {op_done:>2d}/{op_total:<2d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
