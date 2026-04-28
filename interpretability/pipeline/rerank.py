"""Stage A — LLM rerank of cached SERPs.

Inputs (no HTTP):
    geodml_data/data/serp/phase0_top{20,50}_{searxng,ddg}.parquet

Outputs (per run_id, where run_id = "{engine}_{ModelTag}_serp{N}_top{K}_{variant}"):
    geodml_data/data/runs/{run_id}/phase2/keywords.jsonl
    geodml_data/data/runs/{run_id}/phase2/rankings.csv
    geodml_data/data/runs/{run_id}/phase2/.rerank_ckpt.json

Each line of keywords.jsonl mirrors the upstream gather_data.py JSON envelope
plus a few extra fields (``prompt_variant``, ``engine``, ``pool``, ``top_n``,
``run_id``, ``rank_changes``) so ``pipeline.merge`` can consume it without
joining back to the rerank logs.

Ported from:
    pipeline/gather_data.py:_build_domain_url_map
    pipeline/gather_data.py:_attach_urls
    pipeline/gather_data.py:_fallback_extract     (variant-aware, see note)
    pipeline/gather_data.py:rank_domains_with_llm (uses LocalRanker now)
    pipeline/gather_data.py:compute_rank_changes

What changed in the port:

- ``InferenceClient.chat_completion`` -> ``utils.make_ranker(backend="local")``.
  No HTTP. Models load 4-bit on the booster GPUs; HF_HUB_OFFLINE=1 in sbatch.
- Domain parsing: use ``utils.parse_ranked_domains`` (already strips DeepSeek
  ``<think>`` blocks, deduplicates).
- ``_fallback_extract`` is variant-aware: with ``variant="biased"`` it keeps
  the upstream skip list (g2.com, capterra.com, ...) for byte-compatible
  reproduction. With ``variant="neutral"`` it drops the list entirely - just
  takes the first ``top_n`` unique domains in SERP order. Fallback fires only
  when the LLM output cannot be parsed.
- Per-keyword JSONL writes are atomic-append (line-buffered open in 'a' mode).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from interpretability.pipeline import config as C
from interpretability.pipeline.prompts import (
    build_rerank_prompt,
    PromptVariant,
)
from interpretability.utils import (
    Checkpoint,
    _extract_domain,
    data_root,
    load_serp,
    make_ranker,
    parse_ranked_domains,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_domain_url_map(search_results: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for r in search_results:
        url = r.get("url", "")
        d = _extract_domain(url)
        if d and d not in out:
            out[d] = url
    return out


def _attach_urls(domains: list[str], domain_url_map: dict[str, str]) -> list[dict]:
    return [{"domain": d, "url": domain_url_map.get(d, "")} for d in domains]


# Same skip list the upstream gather_data.py used. Editorial — only used on the
# "biased" path so the byte-compat reproduction matches.
_BIASED_FALLBACK_SKIP = {
    "g2.com", "capterra.com", "wikipedia.org", "youtube.com",
    "reddit.com", "quora.com", "forbes.com", "techcrunch.com",
    "gartner.com", "trustradius.com", "softwareadvice.com",
    "getapp.com", "pcmag.com", "techradar.com", "cnet.com",
}


def _fallback_extract(
    search_results: list[dict], top_n: int, variant: PromptVariant,
) -> list[str]:
    skip = _BIASED_FALLBACK_SKIP if variant == "biased" else set()
    out: list[str] = []
    for r in search_results:
        d = _extract_domain(r.get("url", ""))
        if d and d not in out and d not in skip:
            out.append(d)
        if len(out) >= top_n:
            break
    return out


def compute_rank_changes(
    raw_results: list[dict], post_llm_domains: list[str],
) -> list[dict]:
    """Mirror of pipeline/gather_data.py:compute_rank_changes."""
    pre_domains: list[str] = []
    for r in raw_results:
        d = _extract_domain(r.get("url", ""))
        if d and d not in pre_domains:
            pre_domains.append(d)
    pre_rank_map = {d: i + 1 for i, d in enumerate(pre_domains)}

    changes: list[dict] = []
    for post_rank_0, domain in enumerate(post_llm_domains):
        post_rank = post_rank_0 + 1
        pre_rank = pre_rank_map.get(domain)
        rank_delta = (pre_rank - post_rank) if pre_rank is not None else None
        changes.append({
            "domain": domain,
            "pre_rank": pre_rank,
            "post_rank": post_rank,
            "rank_delta": rank_delta,
        })
    return changes


# ─── core rerank ──────────────────────────────────────────────────────────────

def rank_one_keyword(
    keyword: str,
    search_results: list[dict],
    *,
    ranker,
    model_id: str,
    top_n: int,
    variant: PromptVariant,
) -> dict:
    """Re-rank one keyword's SERP candidates and return the full record.

    Equivalent envelope to ``gather_data.rank_domains_with_llm`` plus
    ``compute_rank_changes`` and prompt-variant metadata.
    """
    record = {
        "keyword": keyword,
        "llm_role": "re-ranker (LLM re-orders results by relevance)",
        "llm_model": model_id,
        "llm_parameters": {
            "max_tokens": C.LLM_MAX_TOKENS,
            "temperature": C.LLM_TEMPERATURE,
        },
        "prompt": None,
        "raw_llm_response": None,
        "llm_query_timestamp_utc": None,
        "llm_response_timestamp_utc": None,
        "ranked_domains": [],
        "ranked_results": [],
        "used_fallback": False,
        "error": None,
        "prompt_variant": variant,
        "rank_changes": [],
    }

    if not search_results:
        record["error"] = "no search results provided"
        return record

    domain_url_map = _build_domain_url_map(search_results)
    prompt = build_rerank_prompt(keyword, search_results, top_n=top_n, variant=variant)
    record["prompt"] = prompt
    record["llm_query_timestamp_utc"] = _utcnow_iso()

    try:
        llm_output = ranker.rank(
            prompt,
            max_tokens=C.LLM_MAX_TOKENS,
            temperature=C.LLM_TEMPERATURE,
        )
    except Exception as e:
        record["llm_response_timestamp_utc"] = _utcnow_iso()
        record["error"] = f"{type(e).__name__}: {e}"
        record["used_fallback"] = True
        domains = _fallback_extract(search_results, top_n, variant)
        record["ranked_domains"] = domains
        record["ranked_results"] = _attach_urls(domains, domain_url_map)
        record["rank_changes"] = compute_rank_changes(search_results, domains)
        return record

    record["llm_response_timestamp_utc"] = _utcnow_iso()
    record["raw_llm_response"] = llm_output
    domains = parse_ranked_domains(llm_output)[:top_n]

    # If the LLM returned nothing parseable, fall back rather than emit an
    # empty rank list (mirrors upstream behavior).
    if not domains:
        record["used_fallback"] = True
        domains = _fallback_extract(search_results, top_n, variant)

    record["ranked_domains"] = domains
    record["ranked_results"] = _attach_urls(domains, domain_url_map)
    record["rank_changes"] = compute_rank_changes(search_results, domains)
    return record


# ─── per-keyword IO ───────────────────────────────────────────────────────────

def _serp_to_results(rows: pd.DataFrame) -> list[dict]:
    """phase0 parquet rows -> the dict shape gather_data used."""
    out: list[dict] = []
    for _, r in rows.iterrows():
        out.append({
            "position": int(r["position"]),
            "title":    str(r.get("title", "") or ""),
            "url":      str(r.get("url", "") or ""),
            "snippet":  str(r.get("snippet", "") or ""),
        })
    return out


def _iter_keyword_groups(serp: pd.DataFrame, pool: int) -> Iterable[tuple[str, pd.DataFrame]]:
    """Yield (keyword, top-`pool` rows sorted by position)."""
    for kw, g in serp.groupby("keyword", sort=False):
        g = g.sort_values("position").head(pool)
        yield kw, g


def _flatten_rank_changes(record: dict) -> list[dict]:
    """One row per (keyword, domain) for rankings.csv."""
    out = []
    for c in record.get("rank_changes", []):
        out.append({
            "keyword":     record["keyword"],
            "domain":      c["domain"],
            "url":         next(
                (rr["url"] for rr in record["ranked_results"] if rr["domain"] == c["domain"]),
                "",
            ),
            "pre_rank":    c["pre_rank"],
            "post_rank":   c["post_rank"],
            "rank_delta":  c["rank_delta"],
            "prompt_variant": record["prompt_variant"],
            "llm_model":   record["llm_model"],
        })
    return out


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage A: LLM rerank cached SERPs (no HTTP).",
    )
    ap.add_argument("--engine", default="searxng", choices=C.ENGINES,
                    help="Cached SERP engine (searxng or ddg).")
    ap.add_argument("--pool", type=int, default=50, choices=(20, 50),
                    help="SERP pool size to rerank from (must match a cached parquet).")
    ap.add_argument("--top-n", type=int, default=10,
                    help="LLM-side top-N to ask for.")
    ap.add_argument("--model",
                    default=os.getenv("PRIMARY_MODEL", C.LLM_MODELS[0]),
                    help="HuggingFace model ID for the local 4-bit ranker.")
    ap.add_argument("--backend", choices=("local", "api"),
                    default=os.getenv("RERANK_BACKEND", "local"),
                    help="'local' = LocalRanker on cluster GPU. 'api' = HF Inference API "
                         "(login-node only; needs HF_TOKEN).")
    ap.add_argument("--variant", choices=("biased", "neutral"),
                    default=os.getenv("PROMPT_VARIANT", "biased"),
                    help="Prompt variant. Defaults to PROMPT_VARIANT env or 'biased'.")
    ap.add_argument("--data-root", default=None,
                    help="Override geodml_data root (defaults to GEODML_DATA_ROOT or ./geodml_data).")
    ap.add_argument("--max-keywords", type=int, default=None,
                    help="Cap (smoke testing). None = all keywords.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip keywords already present in the JSONL output.")
    ap.add_argument("--out-run-id", default=None,
                    help="Override the auto-derived run_id (rare; mostly for smoke tests).")
    args = ap.parse_args()

    root = data_root(args.data_root)
    run_id = args.out_run_id or C.run_label_with_variant(
        args.engine, args.model, args.pool, args.top_n, args.variant,
    )
    out_dir = root / "data" / "runs" / run_id / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "keywords.jsonl"
    csv_path   = out_dir / "rankings.csv"
    ckpt_path  = out_dir / ".rerank_ckpt.json"

    # Load checkpoint + figure out which keywords are already done.
    ckpt = Checkpoint.load(ckpt_path)
    done: set[str] = set(ckpt.data.get("seen", []))

    # Also reconcile against the JSONL contents in case the checkpoint was
    # truncated (e.g. node OOM mid-write).
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
        # Truncate outputs on a fresh run.
        jsonl_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
        done = set()

    # Open output handles (line-buffered append).
    jsonl_f = jsonl_path.open("a", buffering=1)
    csv_new = not csv_path.exists()
    csv_f = csv_path.open("a", buffering=1, newline="")
    csv_w = csv.DictWriter(
        csv_f,
        fieldnames=[
            "keyword", "domain", "url",
            "pre_rank", "post_rank", "rank_delta",
            "prompt_variant", "llm_model",
        ],
    )
    if csv_new:
        csv_w.writeheader()

    # Load SERPs + initialize ranker.
    print(f"[rerank] run_id={run_id}", flush=True)
    print(f"[rerank] backend={args.backend} model={args.model} variant={args.variant}", flush=True)

    serp = load_serp(backend=args.engine, pool=args.pool, root=root)
    print(f"[rerank] cached SERP rows={len(serp):,} keywords={serp.keyword.nunique():,}", flush=True)

    ranker = make_ranker(args.backend, args.model)

    n_done = 0
    n_err  = 0
    iter_groups = _iter_keyword_groups(serp, args.pool)

    pbar = tqdm(total=serp["keyword"].nunique(), desc=f"rerank {run_id}", initial=len(done))
    for kw, g in iter_groups:
        if kw in done:
            continue
        if args.max_keywords is not None and n_done >= args.max_keywords:
            break

        results = _serp_to_results(g)

        try:
            rec = rank_one_keyword(
                kw, results,
                ranker=ranker,
                model_id=args.model,
                top_n=args.top_n,
                variant=args.variant,
            )
        except Exception as e:
            n_err += 1
            print(f"[rerank] FATAL keyword={kw!r}: {type(e).__name__}: {e}", flush=True)
            if n_err <= 3:
                traceback.print_exc()
            continue

        # Decorate with run-level metadata (so merge.py is single-source).
        rec.update({
            "engine":   args.engine,
            "pool":     args.pool,
            "top_n":    args.top_n,
            "run_id":   run_id,
        })

        jsonl_f.write(json.dumps(rec, default=str) + "\n")
        for row in _flatten_rank_changes(rec):
            csv_w.writerow(row)

        ckpt.mark(kw)
        n_done += 1
        if n_done % 25 == 0:
            ckpt.save()
        pbar.update(1)

    pbar.close()
    ckpt.save()
    jsonl_f.close()
    csv_f.close()

    print(f"[rerank] done: new={n_done} errors={n_err} total_seen={len(ckpt.data.get('seen', []))}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
