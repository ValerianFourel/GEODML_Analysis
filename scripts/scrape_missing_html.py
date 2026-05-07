#!/usr/bin/env python3
"""Fetch HTML for the URLs that are in phase0 SERP files but not yet cached.

For each (engine, pool):
  1. Read data/serp/phase0_top<pool>_<engine>.parquet
  2. For each URL, compute sha256(url)[:16].html
  3. Check whether that filename exists in each model's html_cache dir
     (data/runs/<engine>_<Model>_serp<pool>_top10/phase2/html_cache/)
  4. If missing in any model's cache, fetch the URL and write to all
     missing locations.

After scraping, invalidate the passage cache parquets so the next rerank
rebuilds them with the new URLs included.

Usage:
    GEODML_DATA_ROOT=... python scripts/scrape_missing_html.py --dry-run
    GEODML_DATA_ROOT=... python scripts/scrape_missing_html.py --max-urls 100
    GEODML_DATA_ROOT=... python scripts/scrape_missing_html.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

try:
    import requests
except ImportError:
    print("ERROR: pip install requests", file=sys.stderr)
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parent.parent

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
ACCEPT = (
    "text/html,application/xhtml+xml,application/xml;q=0.9,"
    "image/avif,image/webp,*/*;q=0.8"
)


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def fetch(url: str, timeout: float) -> tuple[str | None, str | None]:
    try:
        r = requests.get(
            url,
            headers={"User-Agent": UA, "Accept": ACCEPT, "Accept-Language": "en-US,en;q=0.9"},
            timeout=timeout,
            allow_redirects=True,
        )
        if r.status_code != 200:
            return None, f"http_{r.status_code}"
        ct = (r.headers.get("content-type") or "").lower()
        if not any(t in ct for t in ("text/html", "application/xhtml", "text/plain")):
            return None, f"not_html"
        if len(r.text) < 300:
            return None, "too_small"
        # Cap absurdly large pages
        return r.text[:8_000_000], None
    except requests.exceptions.Timeout:
        return None, "timeout"
    except requests.exceptions.SSLError:
        return None, "ssl"
    except requests.exceptions.ConnectionError:
        return None, "conn"
    except requests.exceptions.TooManyRedirects:
        return None, "redirect_loop"
    except Exception as e:
        return None, f"{type(e).__name__}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.getenv("GEODML_DATA_ROOT", str(REPO_ROOT / "geodml_data")))
    ap.add_argument("--engines", nargs="+", default=["searxng", "ddg"])
    ap.add_argument("--pools", nargs="+", type=int, default=[20, 50])
    ap.add_argument("--models", nargs="+",
                    default=["Llama-3.3-70B-Instruct", "Qwen2.5-72B-Instruct"])
    ap.add_argument("--workers", type=int, default=16,
                    help="Concurrent fetchers")
    ap.add_argument("--timeout", type=float, default=15.0,
                    help="Per-request timeout in seconds")
    ap.add_argument("--max-urls", type=int, default=None,
                    help="Cap total URLs (smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Inventory only, don't fetch")
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    print(f"data root        : {root}")
    print(f"engines          : {args.engines}")
    print(f"pools            : {args.pools}")
    print(f"models           : {args.models}")
    print(f"workers          : {args.workers}")
    print(f"timeout          : {args.timeout}s\n")

    # Index every existing cached file across ALL cache dirs (so we can fill
    # gaps via copy before resorting to HTTP).
    print("indexing existing caches…")
    all_cache_dirs: list[Path] = []
    for engine in args.engines:
        for pool in args.pools:
            for model in args.models:
                d = root / "data" / "runs" / f"{engine}_{model}_serp{pool}_top10" / "phase2" / "html_cache"
                d.mkdir(parents=True, exist_ok=True)
                all_cache_dirs.append(d)
    cached_anywhere: dict[str, Path] = {}  # filename → first dir that has it
    for d in all_cache_dirs:
        for f in d.iterdir():
            if f.name.endswith(".html") and f.name not in cached_anywhere:
                cached_anywhere[f.name] = f
    print(f"  {len(cached_anywhere):,} unique HTML files indexed across all caches\n")

    # Build work lists: cheap copies vs HTTP fetches
    copy_work: list[tuple[Path, list[Path]]] = []  # (source_file, missing_dirs)
    fetch_work: list[tuple[str, list[Path]]] = []  # (url, missing_dirs)

    for engine in args.engines:
        for pool in args.pools:
            serp_path = root / "data" / "serp" / f"phase0_top{pool}_{engine}.parquet"
            if not serp_path.exists():
                continue
            urls = pd.read_parquet(serp_path)["url"].dropna().astype(str).unique().tolist()
            cache_dirs = [
                root / "data" / "runs" / f"{engine}_{model}_serp{pool}_top10" / "phase2" / "html_cache"
                for model in args.models
            ]
            n_copy = n_fetch = 0
            for url in urls:
                fname = url_hash(url) + ".html"
                missing_dirs = [d for d in cache_dirs if not (d / fname).exists()]
                if not missing_dirs:
                    continue
                if fname in cached_anywhere:
                    copy_work.append((cached_anywhere[fname], missing_dirs))
                    n_copy += 1
                else:
                    fetch_work.append((url, missing_dirs))
                    n_fetch += 1
            print(f"  {engine}/pool={pool}: {n_copy:>5d} copy-able  +  {n_fetch:>5d} need HTTP  / {len(urls):,} URLs")

    if not (copy_work or fetch_work):
        print("\nnothing missing — all SERP URLs cached.")
        return 0

    print(f"\ncopy-able        : {len(copy_work):,}  (cheap, in-place)")
    print(f"need HTTP fetch  : {len(fetch_work):,}")

    # Dedupe fetch_work by URL — one HTTP request per unique URL even if it
    # shows up in multiple (engine, pool) SERPs.
    fetch_dedup: dict[str, list[Path]] = {}
    for url, dirs in fetch_work:
        if url in fetch_dedup:
            for d in dirs:
                if d not in fetch_dedup[url]:
                    fetch_dedup[url].append(d)
        else:
            fetch_dedup[url] = list(dirs)
    fetch_work = [(u, ds) for u, ds in fetch_dedup.items()]
    print(f"unique URLs to fetch (dedup): {len(fetch_work):,}")

    if args.dry_run:
        print("\n[DRY RUN] inventory only — no copies, no fetches performed.")
        return 0

    # Phase A — copy existing files into missing model dirs (fast, no network)
    if copy_work:
        print(f"\nphase A: copying {len(copy_work):,} existing files into missing dirs…")
        import shutil
        n_copied = 0
        for src, dirs in copy_work:
            for d in dirs:
                try:
                    shutil.copyfile(src, d / src.name)
                    n_copied += 1
                except Exception as e:
                    print(f"  copy fail: {src.name} → {d}: {e}")
        print(f"  copied {n_copied:,} files\n")

    if not fetch_work:
        passage_dir = root / "data" / "passages"
        if passage_dir.exists():
            for f in passage_dir.glob("passages_*.parquet"):
                f.unlink()
                print(f"invalidated passage cache: {f.name}")
        return 0

    if args.max_urls:
        import random
        random.seed(42)
        random.shuffle(fetch_work)
        fetch_work = fetch_work[: args.max_urls]
        print(f"capped HTTP fetches to {len(fetch_work)} (--max-urls)")

    work = fetch_work  # remaining loop uses 'work'

    # Fetch
    n_ok = n_fail = 0
    reasons: dict[str, int] = {}
    started = time.time()

    def do_one(item: tuple[str, list[Path]]):
        url, dirs = item
        html, err = fetch(url, timeout=args.timeout)
        if html is None:
            return False, url, err
        fname = url_hash(url) + ".html"
        for d in dirs:
            try:
                (d / fname).write_text(html, encoding="utf-8", errors="replace")
            except Exception as e:
                return False, url, f"write_{type(e).__name__}"
        return True, url, None

    print(f"\nfetching {len(work):,} URLs with {args.workers} workers ({args.timeout}s timeout)…")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(do_one, w) for w in work]
        for i, fut in enumerate(as_completed(futures)):
            ok, _, err = fut.result()
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                reasons[err] = reasons.get(err, 0) + 1
            if (i + 1) % 100 == 0 or (i + 1) == len(work):
                el = time.time() - started
                rate = (i + 1) / el if el > 0 else 0
                eta = (len(work) - i - 1) / rate / 60 if rate > 0 else float("inf")
                print(f"  {i+1:>6d}/{len(work):<6d}  ok={n_ok:>5d}  fail={n_fail:>5d}  "
                      f"rate={rate:.1f}/s  eta={eta:.1f}min")

    print(f"\nDONE: ok={n_ok}, fail={n_fail} of {len(work)} ({n_ok/len(work)*100:.1f}% success)")
    print("\nFailure reasons (top):")
    for r, c in sorted(reasons.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {r:20s} {c:>5d}")

    # Invalidate passage caches so next rerank rebuilds with the new URLs
    passage_dir = root / "data" / "passages"
    if passage_dir.exists():
        invalidated = 0
        for f in passage_dir.glob("passages_*.parquet"):
            f.unlink()
            invalidated += 1
        if invalidated:
            print(f"\ninvalidated {invalidated} passage cache parquet(s) — next rerank "
                  "will rebuild them with the new URLs included.")

    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
