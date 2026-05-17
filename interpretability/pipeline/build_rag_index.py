"""Build a per-(engine, pool) RAG index for the ``_rag`` prompt variants.

For each cell:
  1. Read all unique URLs and keywords from the cached SERP parquet.
  2. Re-extract page body text (no 800-char truncation; cap raised to
     ``--max-chars-per-page``).
  3. Chunk each page into ``--chunk-size-chars`` overlapping segments.
  4. Embed all chunks AND all keywords with OpenAI's text-embedding-3-small
     (the same family ChatGPT search uses; 1536-dim by default, configurable
     via ``--embedding-dim`` thanks to Matryoshka truncation).
  5. Persist everything under ``data/rag_index/<engine>_top<pool>/`` so the
     rerank script can do retrieval at run time without paying the embedding
     cost again.

Resume: each stage checks for existing outputs and only processes new rows.

Required env: ``OPENAI_API_KEY``. Pipeline reads ``GEODML_DATA_ROOT`` like the
rest of the pipeline.

Cost: ~$0.02/M tokens. ~44K URLs across 4 cells × ~5 chunks × ~200 tokens =
~44M tokens for chunks + a few hundred KB for keywords. Total ~ $1.

Output schema (under ``data/rag_index/<engine>_top<pool>/``):
  full_passages.parquet      url:str, text:str
  chunks.parquet             url:str, chunk_idx:int32, text:str
  chunk_embeddings.npy       float32 (N_chunks, dim)  -- aligned to chunks rows
  keywords.parquet           keyword:str
  keyword_embeddings.npy     float32 (M_keywords, dim)  -- aligned to keywords rows
  meta.json                  build config + provenance + counts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from interpretability.pipeline import config as C
from interpretability.pipeline.chunker import chunk_text
from interpretability.utils import (
    HTMLLoader,
    data_root,
    extract_passage,
    load_serp,
)

# bge-style retrieval prefixes — text-embedding-3-small handles asymmetric
# query/passage internally, but adding the explicit query-side prefix used by
# bge can give a small uplift and matches the convention.
QUERY_PREFIX = ""  # text-embedding-3-small does NOT need a query prefix
PASSAGE_PREFIX = ""

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIM = 1536  # native dim; configurable via --embedding-dim
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_CHARS_PER_PAGE = 50_000
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_CHUNK_MIN_SIZE = 100


def _index_dir(root: Path, engine: str, pool: int) -> Path:
    return root / "data" / "rag_index" / f"{engine}_top{pool}"


def _resolve_html_run_id(engine: str, pool: int, top_n: int = 10) -> str:
    """Pick which model's html_cache to read for body-text re-extraction.

    Both LLM_MODELS scrape the same SERPs, so either cache works. We prefer the
    PRIMARY_MODEL (Llama by default) for consistency with the existing
    ``_build_passage_map`` flow.
    """
    primary = os.getenv("PRIMARY_MODEL", C.LLM_MODELS[0])
    return C.run_label(engine, primary, pool, top_n)


# ─── Stage 1: full-text extraction ───────────────────────────────────────────


def stage_full_passages(
    serp: pd.DataFrame,
    root: Path,
    out_dir: Path,
    engine: str,
    pool: int,
    max_chars: int,
    flush_every: int = 200,
) -> pd.DataFrame:
    """Extract body text for every URL in ``serp`` (cached, resumable)."""
    out_path = out_dir / "full_passages.parquet"
    cached: dict[str, str] = {}
    if out_path.exists():
        df = pd.read_parquet(out_path)
        cached = dict(zip(df["url"].astype(str), df["text"].astype(str)))
        print(f"[rag-index] loaded {len(cached):,} cached body texts from {out_path.name}", flush=True)

    urls = [str(u) for u in serp["url"].dropna().unique().tolist() if u]
    todo = [u for u in urls if u not in cached]

    if not todo:
        print(f"[rag-index] full_passages: all {len(urls):,} URLs already extracted", flush=True)
        return pd.DataFrame({"url": urls, "text": [cached.get(u, "") for u in urls]})

    cell_run_id = _resolve_html_run_id(engine, pool)
    print(
        f"[rag-index] extracting body text from html_cache of {cell_run_id}: "
        f"{len(todo):,} new URLs (of {len(urls):,})",
        flush=True,
    )

    out: dict[str, str] = dict(cached)
    n_missing = 0
    n_empty = 0

    def _flush():
        df_out = pd.DataFrame({"url": list(out.keys()), "text": list(out.values())})
        tmp = out_path.with_suffix(".tmp.parquet")
        df_out.to_parquet(tmp, index=False)
        tmp.replace(out_path)

    n_new = 0
    with HTMLLoader(cell_run_id, root=root) as loader:
        for url in tqdm(todo, desc="extract body text"):
            html = loader.get_html(url)
            if html is None:
                out[url] = ""
                n_missing += 1
            else:
                text = extract_passage(html, max_chars=max_chars)
                out[url] = text
                if not text:
                    n_empty += 1
            n_new += 1
            if n_new % flush_every == 0:
                _flush()
    _flush()
    print(
        f"[rag-index] full_passages done: total={len(urls):,} new={len(todo):,} "
        f"missing_html={n_missing:,} empty_extract={n_empty:,}",
        flush=True,
    )
    return pd.DataFrame({"url": urls, "text": [out.get(u, "") for u in urls]})


# ─── Stage 2: chunking ───────────────────────────────────────────────────────


def stage_chunks(
    full_passages: pd.DataFrame,
    out_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    chunk_min_size: int,
) -> pd.DataFrame:
    """Chunk all body texts. Cached at chunks.parquet (no embedding column)."""
    out_path = out_dir / "chunks.parquet"
    if out_path.exists():
        df = pd.read_parquet(out_path)
        print(f"[rag-index] loaded {len(df):,} cached chunks from {out_path.name}", flush=True)
        return df

    rows: list[dict] = []
    for url, text in tqdm(
        zip(full_passages["url"], full_passages["text"]),
        total=len(full_passages),
        desc="chunk pages",
    ):
        if not text:
            continue
        chunks = chunk_text(text, size=chunk_size, overlap=chunk_overlap, min_size=chunk_min_size)
        for i, ch in enumerate(chunks):
            rows.append({"url": str(url), "chunk_idx": np.int32(i), "text": ch})

    df = pd.DataFrame(rows, columns=["url", "chunk_idx", "text"])
    df.to_parquet(out_path, index=False)
    print(
        f"[rag-index] chunks done: {len(df):,} chunks from {df['url'].nunique():,} URLs "
        f"(mean {len(df)/max(1, df['url'].nunique()):.1f} chunks/url)",
        flush=True,
    )
    return df


# ─── Stage 3: keyword table ──────────────────────────────────────────────────


def stage_keywords(serp: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Persist the unique keyword list for embedding alignment."""
    out_path = out_dir / "keywords.parquet"
    if out_path.exists():
        df = pd.read_parquet(out_path)
        print(f"[rag-index] loaded {len(df):,} cached keywords from {out_path.name}", flush=True)
        return df

    kws = sorted({str(k) for k in serp["keyword"].dropna().unique()})
    df = pd.DataFrame({"keyword": kws})
    df.to_parquet(out_path, index=False)
    print(f"[rag-index] keywords done: {len(df):,} unique keywords", flush=True)
    return df


# ─── Stage 4: embedding via OpenAI ───────────────────────────────────────────


def _embed_batches(
    client,
    texts: list[str],
    model: str,
    dim: int,
    batch_size: int,
    desc: str,
) -> np.ndarray:
    """Call OpenAI embeddings in batches with retry. Returns float32 (N, dim)."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    n = len(texts)
    pbar = tqdm(total=n, desc=desc)
    i = 0
    backoff = 1.0
    total_tokens = 0
    while i < n:
        batch = texts[i : i + batch_size]
        try:
            kwargs = {"input": batch, "model": model}
            if dim != 1536:
                kwargs["dimensions"] = dim
            r = client.embeddings.create(**kwargs)
        except Exception as e:
            print(f"[rag-index] embed error at i={i}: {type(e).__name__}: {e} — sleep {backoff:.1f}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue
        backoff = 1.0
        for j, dat in enumerate(r.data):
            out[i + j] = np.asarray(dat.embedding, dtype=np.float32)
        if hasattr(r, "usage") and r.usage is not None:
            total_tokens += getattr(r.usage, "total_tokens", 0)
        i += len(batch)
        pbar.update(len(batch))
    pbar.close()
    # text-embedding-3 returns L2-normalized vectors when ``dimensions`` is the
    # native one (1536); when truncated, we re-normalize for cosine = dot.
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out = out / norms
    print(f"[rag-index] {desc}: {len(texts):,} texts, {total_tokens:,} tokens", flush=True)
    return out


def stage_embed_chunks(
    chunks_df: pd.DataFrame,
    out_dir: Path,
    model: str,
    dim: int,
    batch_size: int,
) -> np.ndarray:
    out_path = out_dir / "chunk_embeddings.npy"
    if out_path.exists():
        arr = np.load(out_path)
        if arr.shape == (len(chunks_df), dim):
            print(f"[rag-index] loaded chunk embeddings: shape {arr.shape}", flush=True)
            return arr
        print(
            f"[rag-index] chunk embeddings shape mismatch (got {arr.shape}, want "
            f"{(len(chunks_df), dim)}); re-embedding",
            flush=True,
        )

    from openai import OpenAI

    client = OpenAI()  # picks up OPENAI_API_KEY
    texts = chunks_df["text"].astype(str).tolist()
    arr = _embed_batches(client, texts, model, dim, batch_size, desc="embed chunks")
    np.save(out_path, arr)
    return arr


def stage_embed_keywords(
    keywords_df: pd.DataFrame,
    out_dir: Path,
    model: str,
    dim: int,
    batch_size: int,
) -> np.ndarray:
    out_path = out_dir / "keyword_embeddings.npy"
    if out_path.exists():
        arr = np.load(out_path)
        if arr.shape == (len(keywords_df), dim):
            print(f"[rag-index] loaded keyword embeddings: shape {arr.shape}", flush=True)
            return arr

    from openai import OpenAI

    client = OpenAI()
    texts = keywords_df["keyword"].astype(str).tolist()
    arr = _embed_batches(client, texts, model, dim, batch_size, desc="embed keywords")
    np.save(out_path, arr)
    return arr


# ─── Meta + driver ───────────────────────────────────────────────────────────


def write_meta(
    out_dir: Path,
    engine: str,
    pool: int,
    model: str,
    dim: int,
    chunk_size: int,
    chunk_overlap: int,
    chunk_min_size: int,
    max_chars_per_page: int,
    n_urls: int,
    n_chunks: int,
    n_keywords: int,
) -> None:
    meta = {
        "engine": engine,
        "pool": pool,
        "embedding_model": model,
        "embedding_dim": int(dim),
        "chunk_size_chars": int(chunk_size),
        "chunk_overlap_chars": int(chunk_overlap),
        "chunk_min_size_chars": int(chunk_min_size),
        "max_chars_per_page": int(max_chars_per_page),
        "n_urls": int(n_urls),
        "n_chunks": int(n_chunks),
        "n_keywords": int(n_keywords),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[rag-index] wrote meta.json: {meta}", flush=True)


def build_one_cell(
    *,
    engine: str,
    pool: int,
    model: str,
    dim: int,
    chunk_size: int,
    chunk_overlap: int,
    chunk_min_size: int,
    max_chars_per_page: int,
    batch_size: int,
    root: Path,
) -> None:
    out_dir = _index_dir(root, engine, pool)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n══════ engine={engine} pool={pool} → {out_dir} ══════", flush=True)

    serp = load_serp(backend=engine, pool=pool, root=root)
    print(
        f"[rag-index] SERP rows={len(serp):,} unique_urls={serp.url.nunique():,} "
        f"keywords={serp.keyword.nunique():,}",
        flush=True,
    )

    full = stage_full_passages(serp, root, out_dir, engine, pool, max_chars_per_page)
    chunks = stage_chunks(full, out_dir, chunk_size, chunk_overlap, chunk_min_size)
    keywords = stage_keywords(serp, out_dir)

    if "OPENAI_API_KEY" not in os.environ:
        print(
            "\n[rag-index] OPENAI_API_KEY not set — chunks + keywords are persisted "
            "but embedding stage skipped.\nSet OPENAI_API_KEY and re-run with "
            "--resume to fill embeddings.",
            flush=True,
        )
        write_meta(
            out_dir, engine, pool, model, dim, chunk_size, chunk_overlap,
            chunk_min_size, max_chars_per_page, len(full), len(chunks), len(keywords),
        )
        return

    stage_embed_chunks(chunks, out_dir, model, dim, batch_size)
    stage_embed_keywords(keywords, out_dir, model, dim, batch_size)

    write_meta(
        out_dir, engine, pool, model, dim, chunk_size, chunk_overlap,
        chunk_min_size, max_chars_per_page, len(full), len(chunks), len(keywords),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build the RAG index (chunks + embeddings) for one or more (engine, pool) cells.",
    )
    ap.add_argument("--engine", choices=C.ENGINES, default=None,
                    help="Engine to index (omit to do all engines).")
    ap.add_argument("--pool", type=int, choices=(20, 50), default=None,
                    help="Pool size to index (omit to do all pools).")
    ap.add_argument("--embedding-model", default=DEFAULT_MODEL,
                    help=f"OpenAI embedding model (default: {DEFAULT_MODEL}).")
    ap.add_argument("--embedding-dim", type=int, default=DEFAULT_DIM,
                    help="Embedding dim. text-embedding-3-small supports Matryoshka "
                         f"truncation (default: {DEFAULT_DIM}).")
    ap.add_argument("--chunk-size-chars", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--chunk-overlap-chars", type=int, default=DEFAULT_CHUNK_OVERLAP)
    ap.add_argument("--chunk-min-size-chars", type=int, default=DEFAULT_CHUNK_MIN_SIZE)
    ap.add_argument("--max-chars-per-page", type=int, default=DEFAULT_MAX_CHARS_PER_PAGE)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--resume", action="store_true",
                    help="Reuse cached parquets/npys; only do missing work. (Default; flag kept for clarity.)")
    args = ap.parse_args()

    root = data_root(None)
    print(f"[rag-index] data_root = {root}", flush=True)

    engines = [args.engine] if args.engine else C.ENGINES
    pools = [args.pool] if args.pool else [20, 50]

    for engine in engines:
        for pool in pools:
            build_one_cell(
                engine=engine,
                pool=pool,
                model=args.embedding_model,
                dim=args.embedding_dim,
                chunk_size=args.chunk_size_chars,
                chunk_overlap=args.chunk_overlap_chars,
                chunk_min_size=args.chunk_min_size_chars,
                max_chars_per_page=args.max_chars_per_page,
                batch_size=args.batch_size,
                root=root,
            )

    print("\n[rag-index] all done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
