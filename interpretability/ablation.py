"""Option 1 — Input Ablation.

Validates the DML causal estimates by removing a treatment-relevant feature
from the SERP candidate snippets, re-ranking via the HF Inference API, and
measuring the ranking impact.

Runtime: ~2-4 h for N=500 keywords x 5 treatments x 2 models on the HF
Inference API. Resumable via --resume.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from interpretability.utils import (
    ABLATION_RULES,
    Checkpoint,
    InferenceRanker,
    TREATMENT_TO_COL,
    _extract_domain,
    ablate_snippet,
    ablate_t7,
    build_rerank_prompt,
    data_root,
    load_main_table,
    load_serp,
    parse_ranked_domains,
)


TREATMENTS_TO_ABLATE = [
    "T7_source_earned",
    "T5_topical_comp",
    "T3_structured_data_new",
    "T2a_question_headings",
    "T6_freshness",
    "T1b_stats_density",
]


def stratified_keyword_sample(
    df: pd.DataFrame, treatment: str, n: int, rng: random.Random
) -> list[str]:
    """Pick keywords whose SERP candidates have high variance on the treatment."""
    col = TREATMENT_TO_COL[treatment]
    if col not in df.columns:
        return []
    grp = df.groupby("keyword")[col].agg(["std", "var", "count"])
    # Require at least a few candidates and non-degenerate variance.
    grp = grp[(grp["count"] >= 5) & (grp["std"].notna())]
    if df[col].dropna().unique().size <= 2:
        pool = grp[grp["std"] >= 0.25].index.tolist()
    else:
        thr = grp["var"].quantile(0.75)
        pool = grp[grp["var"] >= thr].index.tolist()
    rng.shuffle(pool)
    return pool[:n]


def ablate_results_for_treatment(
    results: list[dict], treatment: str, df_for_kw: pd.DataFrame
) -> list[dict]:
    """Return a modified copy of `results` with the treatment feature ablated."""
    out: list[dict] = []
    if treatment == "T7_source_earned":
        # Prepend "Official vendor page:" to earned-media candidates' titles.
        earned = set(
            df_for_kw.loc[df_for_kw["treat_source_earned"] == 1, "url"].str.lower()
        )
        for r in results:
            r2 = dict(r)
            if (r2.get("url") or "").lower() in earned:
                r2["title"] = ablate_t7(r2.get("title", ""))
            out.append(r2)
        return out

    if treatment == "T5_topical_comp":
        # Strip the keyword tokens from snippet + title to kill topical overlap.
        kw_tokens: set[str] = set()
        if not df_for_kw.empty:
            kw = df_for_kw["keyword"].iloc[0]
            kw_tokens = {t.lower() for t in kw.split() if len(t) >= 3}
        import re as _re

        for r in results:
            r2 = dict(r)
            for field in ("title", "snippet"):
                text = r2.get(field) or ""
                for t in kw_tokens:
                    text = _re.sub(rf"\b{_re.escape(t)}\b", "", text, flags=_re.I)
                r2[field] = _re.sub(r"\s+", " ", text).strip()
            out.append(r2)
        return out

    # Regex-based rule for T1/T2a/T3/T6.
    if treatment in ABLATION_RULES:
        for r in results:
            r2 = dict(r)
            r2["snippet"] = ablate_snippet(r2.get("snippet") or "", treatment)
            out.append(r2)
        return out

    return [dict(r) for r in results]


def rank_positions(ranked_domains: list[str], candidates: list[dict]) -> dict[str, int]:
    """Map url -> post_rank (1 = top). URLs missing from LLM output get NaN rank."""
    # The original pipeline matches LLM domains against candidates by domain.
    # If the same domain appears multiple times in candidates, rank them in
    # their original SERP order.
    dom_to_urls: dict[str, list[str]] = {}
    for c in candidates:
        d = _extract_domain(c["url"])
        dom_to_urls.setdefault(d, []).append(c["url"])

    out: dict[str, int] = {}
    rank = 1
    used_urls: set[str] = set()
    for d in ranked_domains:
        urls = dom_to_urls.get(d, [])
        for u in urls:
            if u in used_urls:
                continue
            out[u] = rank
            used_urls.add(u)
            rank += 1
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-n", type=int, default=500)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--serp-pool", type=int, default=50)
    ap.add_argument("--serp-backend", default="searxng")
    ap.add_argument(
        "--models",
        default=os.getenv(
            "REMOTE_MODELS",
            "meta-llama/Llama-3.3-70B-Instruct,Qwen/Qwen2.5-72B-Instruct",
        ),
    )
    ap.add_argument(
        "--treatments", default=",".join(TREATMENTS_TO_ABLATE),
        help="Comma-separated treatment labels to ablate.",
    )
    ap.add_argument("--data-root", default=None)
    ap.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "output"),
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "ablation_results.csv"
    ckpt = Checkpoint.load(out_dir / "checkpoint_ablation.json")

    print(f"[ablation] loading main table from {data_root(args.data_root)}")
    main_df = load_main_table(args.data_root)
    serp_df = load_serp(
        backend=args.serp_backend, pool=args.serp_pool, root=args.data_root
    )
    serp_by_kw = {k: g.sort_values("position").to_dict("records")
                  for k, g in serp_df.groupby("keyword")}

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    treatments = [t.strip() for t in args.treatments.split(",") if t.strip()]
    rankers = {m: InferenceRanker(model=m) for m in models}

    # Idempotent resume: if CSV exists, skip (keyword, treatment, model) already done.
    done_keys: set[tuple[str, str, str]] = set()
    if args.resume and results_path.exists():
        prev = pd.read_csv(results_path)
        done_keys = set(zip(prev["keyword"], prev["treatment"], prev["model"]))
        print(f"[ablation] resuming: {len(done_keys)} rows already present")

    # For each treatment, stratified keyword sample.
    all_rows: list[dict] = []
    for treatment in treatments:
        kws = stratified_keyword_sample(main_df, treatment, args.sample_n, rng)
        if not kws:
            print(f"[ablation] no keywords for {treatment}, skipping")
            continue
        print(f"[ablation] treatment={treatment}  n_keywords={len(kws)}")
        for kw in tqdm(kws, desc=treatment):
            cand = serp_by_kw.get(kw)
            if not cand:
                continue
            cand = cand[: max(args.serp_pool, 20)]
            kw_rows = main_df[main_df.keyword == kw]

            ablated = ablate_results_for_treatment(cand, treatment, kw_rows)

            for model in models:
                key = (kw, treatment, model)
                if key in done_keys or ckpt.seen("|".join(key)):
                    continue

                try:
                    base_out = rankers[model].rank(
                        build_rerank_prompt(kw, cand, top_n=args.top_n)
                    )
                    abl_out = rankers[model].rank(
                        build_rerank_prompt(kw, ablated, top_n=args.top_n)
                    )
                except Exception as e:
                    print(f"[ablation] API error for ({kw},{treatment},{model}): {e}")
                    continue

                base_ranks = rank_positions(parse_ranked_domains(base_out), cand)
                abl_ranks = rank_positions(parse_ranked_domains(abl_out), ablated)

                for c in cand:
                    url = c["url"]
                    b = base_ranks.get(url)
                    a = abl_ranks.get(url)
                    if b is None and a is None:
                        continue
                    all_rows.append({
                        "keyword": kw,
                        "url": url,
                        "domain": _extract_domain(url),
                        "treatment": treatment,
                        "model": model,
                        "baseline_rank": b,
                        "ablated_rank": a,
                        # Positive = ablation demoted page = feature was promoting it.
                        "ablation_delta": (
                            (b if b is not None else args.top_n + 1)
                            - (a if a is not None else args.top_n + 1)
                        ) * -1,
                    })

                ckpt.mark("|".join(key))
                if len(all_rows) % 10 == 0:
                    _flush(all_rows, results_path, append=results_path.exists())
                    all_rows = []
                    ckpt.save()

    _flush(all_rows, results_path, append=results_path.exists())
    ckpt.save()
    print(f"[ablation] done -> {results_path}")
    return 0


def _flush(rows: list[dict], path: Path, append: bool) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    mode = "a" if append else "w"
    header = not append
    df.to_csv(path, index=False, mode=mode, header=header)


if __name__ == "__main__":
    raise SystemExit(main())
