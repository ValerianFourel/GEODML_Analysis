"""Stage A' — order-sensitivity probe for the LLM reranker.

Tests whether the rerank top-K is invariant to the *input order* of SERP
candidates. For a given (model, engine, pool, variant) cell we permute the
candidate list with a fixed seed and re-run ``rank_one_keyword``. Comparing the
output top-K domain set against the canonical (position-sorted) rerank tells
us whether the LLM is doing real ranking or anchoring on the input order.

Inputs (no HTTP, no new GPU dependencies beyond the main rerank):
    geodml_data/data/serp/phase0_top{20,50}_{searxng,ddg}.parquet

Outputs:
    geodml_data/data/order_probe/{run_id}_seed{S}.jsonl

Each line mirrors the ``keywords.jsonl`` envelope written by ``rerank.py``,
plus three new fields:

  - ``seed`` (int)            : RNG seed used for this run
  - ``input_order_perm``      : list of original positions in the order shown
                                to the LLM (length == pool)
  - ``original_order``        : list[int] 1..pool, for cross-checking

The output schema is otherwise identical to rerank.py so the analyzer can
reuse the same JSONL reader.

Usage:
    python -m interpretability.pipeline.order_probe \
        --variant neutral --model meta-llama/Llama-3.3-70B-Instruct \
        --engine searxng --pool 50 --seed 42 \
        [--max-keywords N] [--resume] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from interpretability.pipeline import config as C
from interpretability.pipeline.rerank import (
    _build_passage_map,
    _build_retrieved_map,
    _iter_keyword_groups,
    _serp_to_results,
    rank_one_keyword,
)
from interpretability.utils import (
    Checkpoint,
    data_root,
    load_serp,
    make_ranker,
)


def _shuffle_for_keyword(results: list[dict], *, seed: int, keyword: str) -> tuple[list[dict], list[int]]:
    """Permute the candidate list with a per-keyword RNG.

    The RNG is keyed on ``(seed, keyword)`` so the same seed gives a stable
    permutation per keyword regardless of iteration order — important so adding
    or removing keywords later does not change the shuffle for unrelated ones.

    Returns ``(shuffled_results, perm)`` where ``perm[i]`` is the ORIGINAL
    1-indexed ``position`` field of the candidate now sitting at index ``i``.
    """
    rng = random.Random(f"{seed}::{keyword}")
    indexed = list(enumerate(results))
    rng.shuffle(indexed)
    shuffled = [r for _, r in indexed]
    perm = [int(r["position"]) for r in shuffled]
    return shuffled, perm


# ─── smoke-mode mock ranker (no GPU) ─────────────────────────────────────────

class _MockRanker:
    """Deterministic mock for ``--smoke``. Returns the candidates that were
    fed in, in the same order, formatted as the rerank parser expects.

    Lets us verify that:
      - the prompt was built from the SHUFFLED list (so the parsed top-K
        reflects the shuffled input order)
      - the JSONL envelope is identical to rerank.py minus the new fields
    """

    def rank(self, prompt: str, **_) -> str:
        # Heuristic: extract every "<n>. [domain] ..." line from the prompt.
        out: list[str] = []
        for line in prompt.splitlines():
            line = line.strip()
            if not line or line[0].isalpha():
                continue
            if "[" in line and "]" in line:
                lb, rb = line.find("["), line.find("]")
                if 0 <= lb < rb:
                    domain = line[lb + 1:rb].strip()
                    if domain and domain not in out:
                        out.append(domain)
        return "\n".join(f"{i + 1}. {d}" for i, d in enumerate(out[:10]))


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage A' — order-sensitivity probe for the LLM reranker.",
    )
    ap.add_argument("--engine", default="searxng", choices=C.ENGINES)
    ap.add_argument("--pool", type=int, default=50, choices=(20, 50))
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--model",
                    default=os.getenv("PRIMARY_MODEL", C.LLM_MODELS[0]),
                    help="HuggingFace model ID for the local 4-bit ranker.")
    ap.add_argument("--backend", choices=("local", "api", "openai"),
                    default=os.getenv("RERANK_BACKEND", "local"))
    ap.add_argument("--precision", choices=("full", "4bit"),
                    default=os.getenv("LOCAL_PRECISION", "full"),
                    help="Local-backend precision. 'full' = bf16 (matches API endpoint). "
                         "'4bit' = nf4 quantization (legacy default). Ignored for "
                         "backend=api/openai.")
    ap.add_argument("--variant",
                    choices=(
                        "biased", "neutral",
                        "biased_passage", "neutral_passage",
                        "biased_rag", "neutral_rag",
                    ),
                    default=os.getenv("PROMPT_VARIANT", "biased"))
    ap.add_argument("--top-k-rag", type=int, default=3,
                    help="Number of retrieved chunks per (keyword, url) for *_rag variants.")
    ap.add_argument("--seed", type=int, required=True,
                    help="RNG seed for the per-keyword candidate shuffle.")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--max-keywords", type=int, default=None,
                    help="Cap (smoke testing). None = all keywords.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip keywords already present in the output JSONL.")
    ap.add_argument("--smoke", action="store_true",
                    help="Use a mock ranker (CPU, no GPU). Implies --max-keywords 5 if not set.")
    ap.add_argument("--out-run-id", default=None,
                    help="Override the auto-derived run_id (rare; mostly for smoke tests).")
    args = ap.parse_args()

    root = data_root(args.data_root)
    run_id = args.out_run_id or C.run_label_with_variant(
        args.engine, args.model, args.pool, args.top_n, args.variant,
    )
    out_dir = root / "data" / "order_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{run_id}_seed{args.seed}.jsonl"
    ckpt_path = out_dir / f".{run_id}_seed{args.seed}_ckpt.json"

    # Resume: rebuild the seen set from the existing JSONL.
    ckpt = Checkpoint.load(ckpt_path)
    done: set[str] = set(ckpt.data.get("seen", []))
    if args.resume and jsonl_path.exists():
        with jsonl_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("keyword"):
                        done.add(rec["keyword"])
                except json.JSONDecodeError:
                    pass
    if not args.resume:
        jsonl_path.unlink(missing_ok=True)
        done = set()

    if args.smoke and args.max_keywords is None:
        args.max_keywords = 5

    print(f"[order_probe] run_id={run_id} seed={args.seed}", flush=True)
    print(f"[order_probe] backend={args.backend} model={args.model} "
          f"variant={args.variant} precision={args.precision} smoke={args.smoke}",
          flush=True)
    print(f"[order_probe] output -> {jsonl_path}", flush=True)

    # Load SERPs.
    serp = load_serp(backend=args.engine, pool=args.pool, root=root)
    print(f"[order_probe] cached SERP rows={len(serp):,} "
          f"keywords={serp.keyword.nunique():,}", flush=True)

    passage_map: dict[str, str] | None = None
    retrieved_map: dict[tuple[str, str], str] | None = None
    if args.variant.endswith("_passage"):
        passage_map = _build_passage_map(
            serp, root, args.engine, args.model, args.pool, args.top_n,
        )
    elif args.variant.endswith("_rag"):
        retrieved_map = _build_retrieved_map(
            serp, root, args.engine, args.pool, k=args.top_k_rag,
        )

    # Ranker (mock under --smoke, real otherwise).
    ranker = (
        _MockRanker() if args.smoke
        else make_ranker(args.backend, args.model, precision=args.precision)
    )

    jsonl_f = jsonl_path.open("a", buffering=1)
    n_done = 0
    n_err = 0

    pbar = tqdm(total=serp["keyword"].nunique(),
                desc=f"order_probe {run_id} s={args.seed}",
                initial=len(done))
    for kw, g in _iter_keyword_groups(serp, args.pool):
        if kw in done:
            continue
        if args.max_keywords is not None and n_done >= args.max_keywords:
            break

        results = _serp_to_results(
            g, passage_map=passage_map,
            keyword=kw, retrieved_map=retrieved_map,
        )
        original_order = [int(r["position"]) for r in results]
        shuffled, perm = _shuffle_for_keyword(results, seed=args.seed, keyword=kw)

        try:
            rec = rank_one_keyword(
                kw, shuffled,
                ranker=ranker,
                model_id=args.model,
                top_n=args.top_n,
                variant=args.variant,
                backend=args.backend,
                precision=args.precision,
            )
        except Exception as e:
            n_err += 1
            print(f"[order_probe] FATAL keyword={kw!r}: {type(e).__name__}: {e}",
                  flush=True)
            if n_err <= 3:
                traceback.print_exc()
            continue

        # Decorate with run-level + experiment-specific metadata.
        rec.update({
            "engine":   args.engine,
            "pool":     args.pool,
            "top_n":    args.top_n,
            "run_id":   run_id,
            "seed":     args.seed,
            "input_order_perm": perm,
            "original_order":   original_order,
        })

        jsonl_f.write(json.dumps(rec, default=str) + "\n")
        ckpt.mark(kw)
        n_done += 1
        if n_done % 25 == 0:
            ckpt.save()
        pbar.update(1)

    pbar.close()
    ckpt.save()
    jsonl_f.close()

    print(f"[order_probe] done: new={n_done} errors={n_err} "
          f"total_seen={len(ckpt.data.get('seen', []))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
