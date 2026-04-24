"""Option 2 — Gradient x Input Saliency.

Identifies which input tokens drive the LLM's ranking decision using
gradient-x-embedding saliency. Runs locally on a 4-bit quantized 8B model.

Hypothesis: saliency ratio (treatment-relevant tokens / other tokens) is
high for treatments with strong DML coefficients (T7, T5).

Runtime: ~1-2 h on A100-40G or RTX 4090 with N=200 examples.
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
    TREATMENT_TO_COL,
    _extract_domain,
    build_rerank_prompt,
    data_root,
    load_main_table,
    load_serp,
    tag_token_for_treatment,
)


def _load_model(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    kw: dict = {"device_map": "auto"} if device == "cuda" else {"torch_dtype": torch.float32}
    if device == "cuda":
        try:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kw["quantization_config"] = quant
        except Exception as e:
            print(f"[saliency] bitsandbytes unavailable ({e}); falling back to fp16")
            kw["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    model.eval()
    return model, tok


def _embed_layer(model):
    """Find the input embedding layer across families (LLaMA, Qwen, ...)."""
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    raise RuntimeError("Could not locate input embedding layer")


def saliency_over_prompt(model, tok, prompt: str, device: str):
    """Return (tokens: list[str], saliency: np.ndarray) of length = input tokens.

    Saliency = ||grad(loss) * embedding|| along the embedding dim, where loss
    is the log-prob of the model's first generated token. This is a cheap
    proxy for "which input tokens drove the ranking decision".
    """
    import torch

    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = enc.input_ids.to(device)
    attn = enc.attention_mask.to(device)

    emb_layer = _embed_layer(model)
    inputs_embeds = emb_layer(input_ids).detach().clone()
    inputs_embeds.requires_grad_(True)

    # Greedy-decode 1 token to get the "decision" logit.
    out = model(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=False)
    logits = out.logits[:, -1, :]
    next_id = logits.argmax(dim=-1)
    loss = -torch.log_softmax(logits, dim=-1).gather(1, next_id.unsqueeze(0)).sum()

    loss.backward()
    grads = inputs_embeds.grad  # (1, seq, hidden)
    # Saliency per token = ||grad * embedding||_2 along hidden dim.
    sal = (grads * inputs_embeds).detach().float().norm(dim=-1).squeeze(0).cpu().numpy()

    toks = tok.convert_ids_to_tokens(input_ids[0].tolist())
    return toks, sal


def sample_keywords(df: pd.DataFrame, n: int, rng: random.Random) -> list[dict]:
    """Balanced sample: half treat_source_earned=1, half =0."""
    pos = df[df["treat_source_earned"] == 1]["keyword"].dropna().unique().tolist()
    neg = df[df["treat_source_earned"] == 0]["keyword"].dropna().unique().tolist()
    rng.shuffle(pos)
    rng.shuffle(neg)
    pool = pos[: n // 2] + neg[: n - n // 2]
    return pool


def aggregate_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Per-treatment mean saliency ratio."""
    rows = []
    for t, g in scores.groupby("treatment"):
        treat = g[g["is_treatment_token"] == 1]["saliency_score"]
        other = g[g["is_treatment_token"] == 0]["saliency_score"]
        if treat.empty or other.empty:
            continue
        rows.append({
            "treatment": t,
            "mean_treatment_saliency": float(treat.mean()),
            "mean_other_saliency": float(other.mean()),
            "saliency_ratio": float(treat.mean() / max(other.mean(), 1e-12)),
            "n_treatment_tokens": int(treat.size),
            "n_other_tokens": int(other.size),
        })
    return pd.DataFrame(rows).sort_values("saliency_ratio", ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-n", type=int, default=200)
    ap.add_argument(
        "--model",
        default=os.getenv("LOCAL_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
    )
    ap.add_argument("--serp-pool", type=int, default=20)
    ap.add_argument("--serp-backend", default="searxng")
    ap.add_argument("--top-n", type=int, default=10)
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
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "saliency_scores.csv"
    summary_path = out_dir / "saliency_summary.csv"
    ckpt = Checkpoint.load(out_dir / "checkpoint_saliency.json")

    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed. Install torch+transformers+bitsandbytes on the GPU box.")
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[saliency] device={device} model={args.model}")
    if device == "cpu":
        print("[saliency] WARNING: running on CPU. ETA ~30-50x slower than GPU.")

    print(f"[saliency] loading main table from {data_root(args.data_root)}")
    main_df = load_main_table(args.data_root)
    serp_df = load_serp(
        backend=args.serp_backend, pool=args.serp_pool, root=args.data_root
    )
    serp_by_kw = {k: g.sort_values("position").to_dict("records")
                  for k, g in serp_df.groupby("keyword")}

    kws = sample_keywords(main_df, args.sample_n, rng)
    print(f"[saliency] keywords sampled: {len(kws)}")

    model, tok = _load_model(args.model, device)
    if device == "cuda":
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"[saliency] GPU memory allocated after load: {mem:.2f} GB")

    done: set[str] = set(ckpt.data.get("seen", []))
    buf: list[dict] = []
    header_written = scores_path.exists() and args.resume

    for kw in tqdm(kws, desc="keywords"):
        if kw in done:
            continue
        cand = serp_by_kw.get(kw)
        if not cand:
            continue
        cand = cand[: args.serp_pool]
        prompt = build_rerank_prompt(kw, cand, top_n=args.top_n)

        try:
            toks, sal = saliency_over_prompt(model, tok, prompt, device)
        except Exception as e:
            print(f"[saliency] skip {kw}: {e}")
            continue

        for i, (t, s) in enumerate(zip(toks, sal)):
            # Strip leading special chars used by BPE (Ġ, ▁) for tagging only.
            clean = t.replace("Ġ", " ").replace("▁", " ").strip()
            if not clean:
                continue
            tags = tag_token_for_treatment(clean)
            if tags:
                for tag in tags:
                    buf.append({
                        "keyword": kw,
                        "token": clean,
                        "pos": i,
                        "saliency_score": float(s),
                        "treatment": tag,
                        "is_treatment_token": 1,
                        "model": args.model,
                    })
            else:
                # Sample ~10% of non-treatment tokens as the "other" baseline
                # to keep the CSV size bounded.
                if rng.random() < 0.1:
                    buf.append({
                        "keyword": kw,
                        "token": clean,
                        "pos": i,
                        "saliency_score": float(s),
                        "treatment": "OTHER",
                        "is_treatment_token": 0,
                        "model": args.model,
                    })

        ckpt.mark(kw)
        if len(buf) >= 1000:
            df = pd.DataFrame(buf)
            df.to_csv(scores_path, mode=("a" if header_written else "w"),
                      header=not header_written, index=False)
            header_written = True
            buf.clear()
            ckpt.save()

    if buf:
        df = pd.DataFrame(buf)
        df.to_csv(scores_path, mode=("a" if header_written else "w"),
                  header=not header_written, index=False)
        header_written = True
    ckpt.save()

    # Build summary: for each treatment, mean saliency over treatment tokens vs others.
    if scores_path.exists():
        scores = pd.read_csv(scores_path)
        # Need both treatment-tagged rows and OTHER rows — broadcast OTHER across
        # every treatment for the ratio calculation.
        treats = [t for t in scores["treatment"].unique() if t != "OTHER"]
        other_rows = scores[scores["treatment"] == "OTHER"]
        expanded = [scores[scores["treatment"] != "OTHER"]]
        for t in treats:
            clone = other_rows.copy()
            clone["treatment"] = t
            expanded.append(clone)
        scores_for_summary = pd.concat(expanded, ignore_index=True)
        summary = aggregate_summary(scores_for_summary)
        summary.to_csv(summary_path, index=False)
        print("[saliency] summary:")
        print(summary.to_string(index=False))

    print(f"[saliency] done -> {scores_path}, {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
