"""Option 3 — Probing Classifiers.

For each transformer layer L of a local proxy model, train a logistic-regression
probe on the hidden states to predict a binarized treatment label. Produces a
"probing curve" of accuracy by layer per treatment.

Hypothesis:
  - T7 (source_earned) decodable from early-mid layers (structural/surface).
  - T5 (topical_comp) peaks later (semantic understanding).

Runtime: ~30-60 min on a modern GPU for N=2000 x 4 treatments.
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
    Checkpoint,
    HTMLLoader,
    RUN_IDS,
    data_root,
    load_main_table,
    page_digest,
)


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
        default=os.getenv("LOCAL_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
    )
    ap.add_argument("--run-filter", default=None,
                    help="If set, restrict to one run_id (saves HTML-loading time).")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--data-root", default=None)
    ap.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "output"),
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    out_dir = Path(args.output_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "probing_results.csv"
    ckpt = Checkpoint.load(out_dir / "checkpoint_probing.json")

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
    df = load_main_table(args.data_root)

    default_run = args.run_filter or "searxng_Qwen2.5-72B-Instruct_serp50_top10"
    if default_run not in RUN_IDS:
        print(f"[probing] WARNING: {default_run} not in known RUN_IDS")

    # Build one digest dataset once, reuse across treatments.
    per_treatment_samples: dict[str, pd.DataFrame] = {}
    for label, (col, kind) in PROBING_TREATMENTS.items():
        sub = sample_balanced(df, col, kind, args.sample_n, rng)
        if sub.empty:
            print(f"[probing] skip {label} — no variance in {col}")
            continue
        per_treatment_samples[label] = sub

    # Union of URLs across all treatments (digests are expensive to build).
    all_sub = pd.concat(per_treatment_samples.values(), ignore_index=True)
    all_sub = all_sub.drop_duplicates(subset=["run_id", "url"])
    digests = build_digest_table(all_sub, data_root(args.data_root), default_run)
    digest_map = dict(zip(digests["url"], digests["digest"]))
    print(f"[probing] digests available: {len(digest_map)}")

    model, tok = _load_model(args.model, device)
    if device == "cuda":
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"[probing] GPU memory after load: {mem:.2f} GB")

    all_rows: list[dict] = []
    for label, sub in per_treatment_samples.items():
        if args.resume and ckpt.seen(label):
            print(f"[probing] skipping {label} (already in checkpoint)")
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
        all_rows.extend(rows)

        # Incremental write so a crash doesn't lose earlier treatments.
        mode = "a" if out_path.exists() else "w"
        pd.DataFrame(rows).to_csv(
            out_path, mode=mode, header=(mode == "w"), index=False
        )
        ckpt.mark(label)
        ckpt.save()

    print(f"[probing] done -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
