"""Option 3 — Probing Classifiers.

For each transformer layer L of a local proxy model, train a logistic-regression
probe on the hidden states to predict a binarized treatment label. Produces a
"probing curve" of accuracy by layer per treatment.

Hypothesis:
  - T7 (source_earned) decodable from early-mid layers (structural/surface).
  - T5 (topical_comp) peaks later (semantic understanding).

Runtime: ~30-60 min on a modern GPU for N=2000 x 4 treatments.

Frames (--frame): probing has two scientifically interesting modes — `full`
(is the treatment encoded anywhere) vs `robust_winners` (is it encoded
specifically among pages the LLM actually picks). `both` (default) runs
each frame end-to-end and tags every row with a `frame` column so the
two probing curves can be plotted on shared axes (Figure C).
Per-frame checkpoints + T7 chunk dirs (suffixed _full / _rw) keep
resumption isolated across frames.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from interpretability._robust_winners import (
    engine_from_serp_backend,
    load_robust_winner_pairs,
    short_model_name,
)
from interpretability.utils import (
    Checkpoint,
    HTMLLoader,
    RUN_IDS,
    _extract_domain,
    build_rerank_prompt_with_spans,
    data_root,
    load_main_table,
    load_serp,
    page_digest,
)


FRAME_SUFFIX = {"full": "_full", "robust_winners": "_rw"}


PROBING_TREATMENTS = {
    "T7_source_earned": ("treat_source_earned", "binary"),
    "T5_topical_comp": ("treat_topical_comp", "median_split"),
    "T2a_question_headings": ("treat_question_headings", "binary"),
    "T6_freshness": ("treat_freshness", "median_split"),
}


def sample_balanced(
    df: pd.DataFrame, col: str, kind: str, n: int, rng: random.Random
) -> pd.DataFrame:
    """Return a sampled DataFrame with a binarized `label` column."""
    d = df[df[col].notna()].copy()
    if kind == "binary":
        d["label"] = d[col].astype(int)
    elif kind == "median_split":
        med = d[col].median()
        d["label"] = (d[col] > med).astype(int)
    else:
        raise ValueError(kind)
    # Need both classes.
    if d["label"].nunique() < 2:
        return pd.DataFrame()
    per = n // 2
    pos = d[d["label"] == 1].sample(min(per, (d["label"] == 1).sum()), random_state=rng.randint(0, 2**32 - 1))
    neg = d[d["label"] == 0].sample(min(per, (d["label"] == 0).sum()), random_state=rng.randint(0, 2**32 - 1))
    out = pd.concat([pos, neg]).sample(frac=1, random_state=rng.randint(0, 2**32 - 1))
    return out.reset_index(drop=True)


def _load_model(model_name: str, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    kw: dict = {"device_map": "auto", "output_hidden_states": True}
    if device == "cuda":
        try:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except Exception:
            kw["torch_dtype"] = torch.float16
    else:
        kw = {"output_hidden_states": True, "torch_dtype": torch.float32}

    model = AutoModel.from_pretrained(model_name, **kw)
    model.eval()
    return model, tok


def _t7_extract_one_prompt(
    model, tok, prompt: str, spans: list[dict], device: str, max_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """One forward pass; return (mean_X, last_X, y, meta) for that prompt's spans."""
    import torch

    enc = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
    ).to(device)
    offsets = enc.offset_mapping[0].tolist()
    model_kwargs = {k: v for k, v in enc.items() if k != "offset_mapping"}
    out = model(**model_kwargs, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=1).squeeze(0)  # (L, T, D)
    T = hs.shape[1]

    means, lasts, labels, meta = [], [], [], []
    for s in spans:
        ls, le = s["line_span"]
        idxs = [
            i for i, (cs, ce) in enumerate(offsets)
            if i < T and cs < le and ce > ls and not (cs == 0 and ce == 0)
        ]
        if not idxs:
            continue
        line_tokens = hs[:, idxs, :]
        means.append(line_tokens.mean(dim=1).float().cpu().numpy())
        lasts.append(hs[:, idxs[-1], :].float().cpu().numpy())
        labels.append(int(s["label"]))
        meta.append({
            "url": s["url"], "domain": s["domain"], "label": int(s["label"]),
        })

    if not means:
        return (np.empty((0, 0, 0)), np.empty((0, 0, 0)),
                np.empty((0,), dtype=int), [])
    return (
        np.stack(means, axis=0),
        np.stack(lasts, axis=0),
        np.asarray(labels, dtype=int),
        meta,
    )


def t7_in_context_hidden_states(
    model, tok, keyword_prompts: list[tuple[str, str, list[dict]]],
    device: str, max_len: int, chunk_dir: Path, ckpt: Checkpoint,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Resumable in-context probe data extraction for T7.

    `keyword_prompts` is a list of (kw, prompt, spans). For each kw not yet
    in the checkpoint, runs one forward pass and writes
    `chunk_dir/<safe_kw>.npz` with mean_X / last_X / y / urls. Marks the kw
    in the checkpoint immediately after the chunk is fsynced.

    On every call (including restarts), all existing chunks are loaded and
    concatenated before returning. So a kill mid-keyword loses at most that
    one keyword.
    """
    import torch

    chunk_dir.mkdir(parents=True, exist_ok=True)
    done_kws = set(ckpt.data.get("t7_kws_done", []))

    with torch.no_grad():
        for kw, prompt, spans in tqdm(keyword_prompts, desc="t7-keywords"):
            if kw in done_kws:
                continue
            try:
                mean_X, last_X, y, meta = _t7_extract_one_prompt(
                    model, tok, prompt, spans, device, max_len
                )
            except Exception as e:
                print(f"[probing] T7 skip {kw!r}: {e}")
                continue
            if y.size == 0:
                # Mark anyway so we don't retry every restart.
                done_kws.add(kw)
                ckpt.data["t7_kws_done"] = sorted(done_kws)
                ckpt.save()
                continue
            chunk_path = chunk_dir / f"{_safe_kw(kw)}.npz"
            tmp = chunk_path.with_suffix(".npz.tmp")
            np.savez(
                tmp,
                mean_X=mean_X, last_X=last_X, y=y,
                urls=np.array([m["url"] for m in meta], dtype=object),
                domains=np.array([m["domain"] for m in meta], dtype=object),
            )
            tmp.replace(chunk_path)
            done_kws.add(kw)
            ckpt.data["t7_kws_done"] = sorted(done_kws)
            ckpt.save()

    return _t7_concat_chunks(chunk_dir)


def _safe_kw(kw: str) -> str:
    """Filesystem-safe keyword filename. md5 keeps it short and unique."""
    import hashlib
    h = hashlib.md5(kw.encode("utf-8")).hexdigest()[:16]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in kw)[:40]
    return f"{safe}_{h}"


def _t7_concat_chunks(chunk_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Load all .npz chunks in chunk_dir and concat into a single dataset."""
    means, lasts, ys, meta = [], [], [], []
    for p in sorted(chunk_dir.glob("*.npz")):
        try:
            z = np.load(p, allow_pickle=True)
        except Exception as e:
            print(f"[probing] T7 chunk {p.name} unreadable ({e}); skipping")
            continue
        means.append(z["mean_X"])
        lasts.append(z["last_X"])
        ys.append(z["y"])
        for u, d, lab in zip(z["urls"], z["domains"], z["y"]):
            meta.append({"url": str(u), "domain": str(d), "label": int(lab)})
    if not means:
        return (np.empty((0, 0, 0)), np.empty((0, 0, 0)),
                np.empty((0,), dtype=int), [])
    return (
        np.concatenate(means, axis=0),
        np.concatenate(lasts, axis=0),
        np.concatenate(ys, axis=0),
        meta,
    )


def hidden_states_for_texts(
    model, tok, texts: list[str], device: str, max_len: int = 512, batch_size: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Return (last_token: [N, L, D], mean_pool: [N, L, D]) stacks."""
    import torch

    all_last: list[np.ndarray] = []
    all_mean: list[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="hidden-states"):
            batch = texts[i:i + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            out = model(**enc, output_hidden_states=True)
            # tuple of (num_layers+1) tensors of shape (B, T, D)
            hs = torch.stack(out.hidden_states, dim=1)  # (B, L, T, D)
            mask = enc.attention_mask.unsqueeze(1).unsqueeze(-1).to(hs.dtype)  # (B, 1, T, 1)

            # last non-pad token per row
            idx = enc.attention_mask.sum(dim=1) - 1  # (B,)
            B, L, T, D = hs.shape
            gather_idx = idx.view(B, 1, 1, 1).expand(B, L, 1, D)
            last = hs.gather(2, gather_idx).squeeze(2)  # (B, L, D)

            # masked mean over tokens
            summed = (hs * mask).sum(dim=2)
            counts = mask.sum(dim=2).clamp(min=1)
            mean = summed / counts  # (B, L, D)

            all_last.append(last.float().cpu().numpy())
            all_mean.append(mean.float().cpu().numpy())

    return np.concatenate(all_last, axis=0), np.concatenate(all_mean, axis=0)


def train_probes(
    X: np.ndarray, y: np.ndarray, pooling: str, treatment: str, rng: np.random.Generator
) -> list[dict]:
    """Fit one logistic probe per layer; return list of result rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    rows = []
    n_layers = X.shape[1]
    for layer in range(n_layers):
        Xl = X[:, layer, :]
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xl, y, test_size=0.2, stratify=y, random_state=int(rng.integers(0, 2**31 - 1))
        )
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        clf = LogisticRegression(max_iter=1000, C=0.1, n_jobs=1)
        clf.fit(X_tr, y_tr)
        p_te = clf.predict_proba(X_te)[:, 1]
        y_pred = (p_te >= 0.5).astype(int)
        rows.append({
            "treatment": treatment,
            "layer": layer,
            "pooling": pooling,
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "roc_auc": float(roc_auc_score(y_te, p_te)),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
        })
    return rows


def build_digest_table(
    df: pd.DataFrame, data_root_path: Path, run_filter: str | None
) -> pd.DataFrame:
    """For each row (url, run_id), load HTML and produce a digest. Drop rows
    whose HTML is not available in the per-run cache."""
    # Prefer one run per URL. If the main table has a `run_id` column use it.
    if run_filter is not None:
        df = df[df["run_id"] == run_filter]

    loaders: dict[str, HTMLLoader] = {}
    digests: list[str | None] = []
    for rid, url in tqdm(list(zip(df["run_id"], df["url"])), desc="html->digest"):
        if rid not in loaders:
            loaders[rid] = HTMLLoader(rid, root=data_root_path)
        try:
            html = loaders[rid].get_html(url)
        except FileNotFoundError:
            html = None
        digests.append(page_digest(html) if html else None)
    for l in loaders.values():
        l.close()
    df = df.copy()
    df["digest"] = digests
    return df[df["digest"].notna() & (df["digest"].str.len() > 100)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-n", type=int, default=2000)
    ap.add_argument(
        "--model",
        default=os.getenv("PRIMARY_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        help="Default is the ORIGINAL experiment model. Forward-only, so "
             "~42 GB VRAM in 4-bit. Fits on H100-80G or 2x A100-40G.",
    )
    ap.add_argument(
        "--proxy", action="store_true",
        help="Shortcut: use $PROXY_MODEL (8B) for dev only.",
    )
    ap.add_argument("--run-filter", default=None,
                    help="If set, restrict to one run_id (saves HTML-loading time).")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    # T7 uses an in-context probe (full re-rank prompt + per-URL line spans)
    # because T7 is a domain-list label — the ranker only ever sees [domain],
    # never the page body, so probing body digests is the wrong question.
    ap.add_argument("--t7-keywords", type=int, default=200,
                    help="Number of keywords for the T7 in-context probe.")
    ap.add_argument("--t7-max-len", type=int, default=2048,
                    help="Max prompt length (tokens) for T7 in-context probe.")
    ap.add_argument("--t7-serp-pool", type=int, default=20,
                    help="Candidates per prompt for T7 in-context probe.")
    ap.add_argument("--t7-serp-backend", default="searxng",
                    help="searxng or ddg — SERP source for T7 in-context probe.")
    ap.add_argument("--data-root", default=None)
    ap.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "output"),
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--frame", choices=("full", "robust_winners", "both"), default="both",
        help="Sample frame. 'robust_winners' restricts the probing sample to "
             "(keyword, url) pairs the LLM picks in top-10 under both pools — "
             "tests whether the treatment is encoded ON pages the model uses, "
             "not just anywhere. 'both' runs each frame end-to-end and tags "
             "every row with a `frame` column so the curves share Figure C.",
    )
    args = ap.parse_args()

    if args.proxy:
        args.model = os.getenv("PROXY_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        print(f"[probing] --proxy: using {args.model} (DEV ONLY; NOT for paper)")

    out_dir = Path(args.output_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "probing_results.csv"

    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed. Install torch+transformers+bitsandbytes on the GPU box.")
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[probing] device={device} model={args.model}")
    if device == "cpu":
        print("[probing] WARNING: CPU is slow. ETA ~20x slower than GPU.")

    print(f"[probing] loading main table from {data_root(args.data_root)}")
    df_full = load_main_table(args.data_root)

    default_run = args.run_filter or "searxng_Qwen2.5-72B-Instruct_serp50_top10"
    if default_run not in RUN_IDS:
        print(f"[probing] WARNING: {default_run} not in known RUN_IDS")

    frames = ["full", "robust_winners"] if args.frame == "both" else [args.frame]
    pairs = load_robust_winner_pairs() if "robust_winners" in frames else None
    engine = engine_from_serp_backend(args.t7_serp_backend)
    short_model = short_model_name(args.model)

    model, tok = _load_model(args.model, device)
    if device == "cuda":
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"[probing] GPU memory after load: {mem:.2f} GB")

    for frame in frames:
        suffix = FRAME_SUFFIX[frame]
        ckpt = Checkpoint.load(out_dir / f"checkpoint_probing{suffix}.json")

        print(f"\n[probing] === frame={frame} ===")

        if frame == "robust_winners":
            p = pairs[(pairs["search_engine"] == engine)
                      & (pairs["llm_model"] == short_model)]
            allowed_pairs = set(zip(p["keyword"], p["url"]))
            allowed_kws = set(p["keyword"].unique())
            if not allowed_pairs:
                print(f"[probing] no robust pairs for {engine}+{short_model}; "
                      "skipping frame.")
                continue
            df = df_full[df_full.apply(
                lambda r: (r["keyword"], r["url"]) in allowed_pairs, axis=1
            )].copy()
            print(f"[probing] robust pairs for {engine}+{short_model}: "
                  f"{len(allowed_pairs)} pairs over {len(allowed_kws)} keywords; "
                  f"main_df rows after filter: {len(df)}")
        else:
            df = df_full
            allowed_kws = None

        rng = random.Random(args.seed)
        np_rng = np.random.default_rng(args.seed)

        # T7 takes the in-context path (no body-digest sampling needed).
        digest_treatments = {
            k: v for k, v in PROBING_TREATMENTS.items() if k != "T7_source_earned"
        }

        per_treatment_samples: dict[str, pd.DataFrame] = {}
        for label, (col, kind) in digest_treatments.items():
            sub = sample_balanced(df, col, kind, args.sample_n, rng)
            if sub.empty:
                print(f"[probing] skip {label} — no variance in {col}")
                continue
            per_treatment_samples[label] = sub

        if per_treatment_samples:
            all_sub = pd.concat(per_treatment_samples.values(), ignore_index=True)
            all_sub = all_sub.drop_duplicates(subset=["run_id", "url"])
            digests = build_digest_table(all_sub, data_root(args.data_root), default_run)
            digest_map = dict(zip(digests["url"], digests["digest"]))
            print(f"[probing] digests available: {len(digest_map)}")
        else:
            digest_map = {}

        # T7 in-context: pick keywords whose SERP candidate pool has both earned
        # and non-earned URLs, so each prompt yields informative probe examples.
        t7_prompts: list[tuple[str, str, list[dict]]] = []
        if "T7_source_earned" in PROBING_TREATMENTS:
            serp_df = load_serp(
                backend=args.t7_serp_backend, pool=args.t7_serp_pool,
                root=args.data_root,
            )
            serp_by_kw = {k: g.sort_values("position").to_dict("records")
                          for k, g in serp_df.groupby("keyword")}
            earned_set = {
                _extract_domain(u)
                for u in df_full.loc[df_full["treat_source_earned"] == 1, "url"]
                                .dropna().unique()
            }
            print(f"[probing] T7 earned domains: {len(earned_set)}")

            mixed_kws: list[str] = []
            for kw, results in serp_by_kw.items():
                if allowed_kws is not None and kw not in allowed_kws:
                    continue
                top = results[: args.t7_serp_pool]
                # In robust mode, only count earned URLs that are robust winners.
                if frame == "robust_winners":
                    top = [r for r in top if (kw, r["url"]) in allowed_pairs]
                    if not top:
                        continue
                n_earned = sum(
                    1 for r in top if _extract_domain(r["url"]) in earned_set
                )
                if 0 < n_earned < len(top):
                    mixed_kws.append(kw)
            mixed_kws.sort()
            rng.shuffle(mixed_kws)
            chosen = mixed_kws[: args.t7_keywords]
            print(f"[probing] T7 keywords: {len(chosen)} (out of {len(mixed_kws)} mixed)")

            for kw in chosen:
                top = serp_by_kw[kw][: args.t7_serp_pool]
                if frame == "robust_winners":
                    top = [r for r in top if (kw, r["url"]) in allowed_pairs]
                prompt, spans = build_rerank_prompt_with_spans(kw, top, top_n=10)
                for s in spans:
                    s["label"] = int(s["domain"] in earned_set)
                t7_prompts.append((kw, prompt, spans))

        all_rows: list[dict] = []

        # T7 first, while VRAM is freshest (full prompts can be ~2k tokens).
        # Per-keyword chunked extraction: a kill loses at most one keyword.
        if t7_prompts and not (args.resume and ckpt.seen("T7_source_earned")):
            chunk_dir = out_dir / f"t7_chunks{suffix}"
            mean_X, last_X, y, _meta = t7_in_context_hidden_states(
                model, tok, t7_prompts, device,
                max_len=args.t7_max_len, chunk_dir=chunk_dir, ckpt=ckpt,
            )
            if len(y) >= 100 and y.sum() >= 10 and (len(y) - y.sum()) >= 10:
                print(f"[probing] T7 in-context: n={len(y)}  "
                      f"pos_frac={y.mean():.3f}  hidden={mean_X.shape}")
                rows = []
                rows += train_probes(last_X, y, "last_token", "T7_source_earned", np_rng)
                rows += train_probes(mean_X, y, "mean", "T7_source_earned", np_rng)
                for r in rows:
                    r["frame"] = frame
                all_rows.extend(rows)
                mode = "a" if out_path.exists() else "w"
                pd.DataFrame(rows).to_csv(
                    out_path, mode=mode, header=(mode == "w"), index=False
                )
                ckpt.mark("T7_source_earned")
                ckpt.save()
            else:
                print(f"[probing] T7 in-context: insufficient data (n={len(y)}, "
                      f"pos={int(y.sum())}); skipping")

        for label, sub in per_treatment_samples.items():
            if args.resume and ckpt.seen(label):
                print(f"[probing] skipping {label} (already in {frame} checkpoint)")
                continue

            sub = sub[sub["url"].isin(digest_map)]
            if len(sub) < 100:
                print(f"[probing] {label}: only {len(sub)} rows with digests, skipping")
                continue

            texts = [digest_map[u] for u in sub["url"]]
            y = sub["label"].astype(int).to_numpy()
            print(f"[probing] {label}: n={len(y)}  pos_frac={y.mean():.2f}")

            last_X, mean_X = hidden_states_for_texts(
                model, tok, texts, device,
                max_len=args.max_len, batch_size=args.batch_size,
            )
            print(f"[probing] {label}: hidden-states shape last={last_X.shape}  mean={mean_X.shape}")

            rows = []
            rows += train_probes(last_X, y, "last_token", label, np_rng)
            rows += train_probes(mean_X, y, "mean", label, np_rng)
            for r in rows:
                r["frame"] = frame
            all_rows.extend(rows)

            mode = "a" if out_path.exists() else "w"
            pd.DataFrame(rows).to_csv(
                out_path, mode=mode, header=(mode == "w"), index=False
            )
            ckpt.mark(label)
            ckpt.save()

        print(f"[probing] frame={frame} done")

    print(f"[probing] all frames done -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
