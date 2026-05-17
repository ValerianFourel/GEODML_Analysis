"""Backfill the new ``llm_parameters.{backend,precision}`` fields into
historical JSONL records that pre-date the precision-tracking change.

Why this exists
---------------
Until 2026-05-17 the rerank/order_probe records carried only
``llm_parameters = {max_tokens, temperature}``. We now stratify on
``llm_precision`` (and ``llm_backend``) in the Stage C parquet and in the
HuggingFace dataset push so consumers can distinguish e.g. 4-bit nf4
cluster runs from full-precision API runs from full-precision bf16
cluster runs (the new default).

Without backfill, existing records would show up as ``llm_precision=NULL``
in the parquet and the precision column on HF would be 60% missing.

Heuristic
---------
For each JSONL we infer (backend, precision) by combining:

  * the variant suffix from the path:
      - ``_rag`` → backend=api, precision=api-hf   (created in May 2026 via
        ``scripts/finish_via_api.sh`` with BACKEND=hf, which routes to HF
        Inference and matches the api-hf label).
      - ``_passage`` → backend=api, precision=api-hf  (same path).
      - bare ``biased`` / ``neutral`` → backend=local, precision=4bit-nf4
        (created on JUWELS booster pre-2026-05-17 via run_rerank.sbatch
        which defaulted LocalRanker to quantize=True).
  * the modification time of the JSONL: if the path matches one of those
    rules AND the mtime is BEFORE the precision-tracking switchover
    (default 2026-05-17 00:00 UTC), backfill. Newer files are assumed
    correct (they will be written by the new code path).

If you want to override the inference for a specific file or directory,
use the ``--force-backend`` / ``--force-precision`` flags.

Usage
-----
Dry-run (no writes, just report):
    python scripts/backfill_precision.py \
        --root ~/Hamburg/geodml-dataset --dry-run

Apply in-place (atomic per file via .tmp + rename):
    python scripts/backfill_precision.py --root ~/Hamburg/geodml-dataset

Sanity-check after applying:
    python scripts/backfill_precision.py --root ~/Hamburg/geodml-dataset \
        --dry-run --report-only

Idempotent: records that already have ``llm_parameters.precision`` are
left untouched.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path


# Switchover instant: any JSONL written after this is assumed to already
# carry the precision field. (rerank.py / order_probe.py started writing
# llm_parameters.precision on commit ?? — 2026-05-17.)
DEFAULT_SWITCHOVER = "2026-05-17T00:00:00+00:00"


def infer_from_path(path: Path) -> tuple[str, str] | None:
    """Return (backend, precision_label) for a JSONL path, or None if unknown.

    The path is one of:
      data/runs/<cell>_<variant>/phase2/keywords.jsonl
      data/order_probe/<cell>_<variant>_seed<N>.jsonl

    We look at the trailing variant token.
    """
    name = path.name
    parent = path.parent.name

    # rerank: cell dir name carries the variant
    if name == "keywords.jsonl":
        cell = path.parent.parent.name  # data/runs/<cell>/phase2/
    else:
        # order_probe: filename is <cell>_<variant>_seed<N>.jsonl
        cell = name.rsplit("_seed", 1)[0] if "_seed" in name else name

    # Longest-first match so _passage / _rag win over the bare biased/neutral.
    for suffix, regime in (
        ("_biased_rag",     ("api", "api-hf")),
        ("_neutral_rag",    ("api", "api-hf")),
        ("_biased_passage", ("api", "api-hf")),
        ("_neutral_passage",("api", "api-hf")),
        ("_biased",         ("local", "4bit-nf4")),
        ("_neutral",        ("local", "4bit-nf4")),
    ):
        if cell.endswith(suffix):
            return regime
    return None


def backfill_one(
    path: Path,
    backend: str,
    precision: str,
    *,
    dry_run: bool,
    report_only: bool,
) -> dict:
    """Patch one JSONL in place. Returns counters for the summary."""
    counters = {
        "total":          0,
        "already_set":    0,
        "patched":        0,
        "no_llm_params":  0,
    }
    new_lines: list[str] = []
    changed = False
    with path.open() as f:
        for line in f:
            counters["total"] += 1
            if not line.strip():
                new_lines.append(line)
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Preserve unparseable lines verbatim.
                new_lines.append(line)
                continue
            llm_params = rec.get("llm_parameters")
            if llm_params is None:
                # No llm_parameters at all — rare; backfill the whole dict.
                if not report_only:
                    rec["llm_parameters"] = {
                        "backend": backend,
                        "precision": precision,
                    }
                counters["no_llm_params"] += 1
                counters["patched"] += 1
                changed = True
            elif "precision" in llm_params and llm_params.get("precision"):
                counters["already_set"] += 1
            else:
                if not report_only:
                    llm_params["backend"] = backend
                    llm_params["precision"] = precision
                    rec["llm_parameters"] = llm_params
                counters["patched"] += 1
                changed = True
            new_lines.append(json.dumps(rec, default=str) + "\n")

    if changed and not dry_run and not report_only:
        tmp = path.with_suffix(path.suffix + ".bfp_tmp")
        with tmp.open("w") as f:
            f.writelines(new_lines)
        tmp.replace(path)

    return counters


def collect_jsonls(root: Path) -> list[Path]:
    paths: list[Path] = []
    runs_dir = root / "data" / "runs"
    op_dir   = root / "data" / "order_probe"
    if runs_dir.is_dir():
        paths.extend(sorted(runs_dir.glob("*/phase2/keywords.jsonl")))
    if op_dir.is_dir():
        paths.extend(sorted(op_dir.glob("*.jsonl")))
    # Filter out hidden checkpoint sidecars.
    return [p for p in paths if not p.name.startswith(".")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True,
                    help="Dataset root — the directory containing data/runs/ and "
                         "data/order_probe/. Typically $GEODML_DATA_ROOT.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't write anything; just report what would change.")
    ap.add_argument("--report-only", action="store_true",
                    help="Same as --dry-run but also skip the in-memory record "
                         "patching. Use to summarize existing state.")
    ap.add_argument("--switchover-utc", default=DEFAULT_SWITCHOVER,
                    help=f"ISO-8601 UTC instant; files modified after this are "
                         f"considered already-current and skipped. "
                         f"Default: {DEFAULT_SWITCHOVER}")
    ap.add_argument("--force-backend", default=None,
                    help="Override inferred backend for ALL files in scope.")
    ap.add_argument("--force-precision", default=None,
                    help="Override inferred precision label for ALL files.")
    ap.add_argument("--include-recent", action="store_true",
                    help="Process files even if their mtime is after the "
                         "switchover. Use with --force-precision to relabel "
                         "newly-written records.")
    args = ap.parse_args()

    if args.report_only:
        args.dry_run = True

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: --root not a directory: {root}", file=sys.stderr)
        return 2

    switchover = _dt.datetime.fromisoformat(args.switchover_utc).timestamp()

    jsonls = collect_jsonls(root)
    if not jsonls:
        print(f"No JSONLs found under {root}/data/{{runs,order_probe}}/", file=sys.stderr)
        return 1

    print(f"Scanning {len(jsonls)} JSONLs under {root}")
    print(f"Switchover instant: {args.switchover_utc} "
          f"(files modified after this are skipped unless --include-recent)")
    if args.dry_run:
        print("[DRY-RUN] no files will be modified")
    print()

    totals = {
        "files_scanned":  0,
        "files_skipped":  0,
        "files_patched":  0,
        "records_total":  0,
        "records_patched": 0,
        "records_already_set": 0,
        "records_no_params": 0,
    }
    per_regime: dict[str, int] = {}
    skipped_reasons: dict[str, int] = {}

    for path in jsonls:
        totals["files_scanned"] += 1
        mtime = path.stat().st_mtime
        regime_inferred = infer_from_path(path)
        if regime_inferred is None and not (args.force_backend and args.force_precision):
            skipped_reasons["unknown_variant"] = skipped_reasons.get("unknown_variant", 0) + 1
            totals["files_skipped"] += 1
            continue

        backend = args.force_backend or regime_inferred[0]
        precision = args.force_precision or regime_inferred[1]

        if mtime > switchover and not args.include_recent:
            skipped_reasons["after_switchover"] = skipped_reasons.get("after_switchover", 0) + 1
            totals["files_skipped"] += 1
            continue

        c = backfill_one(
            path, backend, precision,
            dry_run=args.dry_run, report_only=args.report_only,
        )
        totals["records_total"]       += c["total"]
        totals["records_patched"]     += c["patched"]
        totals["records_already_set"] += c["already_set"]
        totals["records_no_params"]   += c["no_llm_params"]
        if c["patched"] > 0:
            totals["files_patched"] += 1
        per_regime[f"{backend}/{precision}"] = per_regime.get(
            f"{backend}/{precision}", 0
        ) + c["patched"]

        label = f"  {path.relative_to(root)}  → {backend}/{precision}"
        action = "would patch" if args.dry_run else "patched"
        if c["patched"] > 0:
            print(f"{label}   ({action} {c['patched']}/{c['total']}, "
                  f"already_set={c['already_set']})")
        elif c["already_set"] == c["total"]:
            pass  # quiet: file is already complete
        else:
            print(f"{label}   (no changes; total={c['total']}, "
                  f"already_set={c['already_set']})")

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    for k, v in totals.items():
        print(f"  {k:<25s}: {v:,}")
    print(f"  patched_by_regime    : {per_regime}")
    print(f"  skipped_reasons      : {skipped_reasons}")
    if args.dry_run and totals["records_patched"] > 0:
        print()
        print("Re-run without --dry-run to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
