"""Option 4 — Weight-space analysis on the ORIGINAL experiment LLMs.

Two analyses that directly probe the trained weights of Llama-3.3-70B /
Qwen2.5-72B (the models whose behaviour the DML coefficients describe):

  A. Logit lens — at each transformer layer, project the hidden state at
     the decision position through the output embedding (lm_head). Track
     when "domain-like" tokens (TLDs, brand words) first dominate the
     distribution. This tells us at what depth the re-ranking decision
     crystallises.

  B. Attention-head importance for T7 — for each (layer, head), measure
     how much attention the decision token pays to URL/domain tokens in
     the prompt. Rank heads. This identifies the specific heads that
     carry the source-type signal that drives the T7 coefficient
     (−1.700 on rank_delta).

Runtime: ~30-90 min for N=100 on Llama-3.3-70B on an H100-80G.
Forward-pass only; no backward, so gradient checkpointing is not needed.
"""

from __future__ import annotations

import argparse
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from interpretability.utils import (
    Checkpoint,
    build_rerank_prompt,
    data_root,
    load_main_table,
    load_serp,
)


# Heuristics for "domain-like" vocab tokens.
DOMAIN_TLD_RE = re.compile(r"\.(com|io|org|net|co|ai|app|dev)$", re.I)
DOMAIN_HINT_RE = re.compile(r"^\.?(com|io|org|net|co|ai|app|dev)$", re.I)
URL_MARK_RE = re.compile(r"(https?://|www\.|\.com|\.io|\.org|\.net)", re.I)


def _load_model(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    kw: dict = {
        "device_map": "auto",
        "output_hidden_states": True,
        "output_attentions": True,
    }
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
        kw.pop("device_map", None)
        kw["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    model.eval()
    return model, tok


def _domain_token_ids(tok) -> np.ndarray:
    """IDs of vocab tokens that look like URL/domain pieces."""
    vocab = tok.get_vocab()  # token_str -> id
    ids: list[int] = []
    for s, i in vocab.items():
        clean = s.replace("Ġ", "").replace("▁", "").lower()
        if not clean:
            continue
        if DOMAIN_TLD_RE.search(clean) or DOMAIN_HINT_RE.match(clean):
            ids.append(i)
    return np.array(sorted(set(ids)), dtype=np.int64)


def _url_positions(input_ids, tok) -> list[int]:
    """Indices of tokens that belong to a URL span in the prompt."""
    toks = tok.convert_ids_to_tokens(input_ids.tolist())
    positions: list[int] = []
    for i, t in enumerate(toks):
        clean = t.replace("Ġ", " ").replace("▁", " ")
        if URL_MARK_RE.search(clean):
            positions.append(i)
    return positions


def sample_keywords(df: pd.DataFrame, n: int, rng: random.Random) -> list[str]:
    """Balanced sample over T7_source_earned so URLs span earned + brand pages."""
    pos = df[df["treat_source_earned"] == 1]["keyword"].dropna().unique().tolist()
    neg = df[df["treat_source_earned"] == 0]["keyword"].dropna().unique().tolist()
    rng.shuffle(pos)
    rng.shuffle(neg)
    return pos[: n // 2] + neg[: n - n // 2]


def run_one(model, tok, prompt: str, device: str, max_len: int,
            domain_ids: np.ndarray) -> dict:
    """One forward pass; return per-layer logit-lens probs + per-head URL attention."""
    import torch

    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    input_ids = enc.input_ids[0]

    with torch.no_grad():
        out = model(
            **enc,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )

    # hidden_states: tuple of (n_layers+1) tensors of shape (1, T, H).
    # The first element is the embedding output (layer 0).
    W_out = model.get_output_embeddings().weight  # (V, H)
    # Optional final norm — apply it before the lm_head for correctness on Llama.
    try:
        final_norm = model.model.norm  # Llama / Qwen family
    except AttributeError:
        final_norm = None

    logit_lens_rows: list[dict] = []
    decision_pos = input_ids.shape[0] - 1
    dom_ids_t = torch.tensor(domain_ids, device=W_out.device, dtype=torch.long)

    for layer, h in enumerate(out.hidden_states):
        vec = h[0, decision_pos, :].to(W_out.dtype)
        if final_norm is not None and layer == len(out.hidden_states) - 1:
            # only the final layer expects the final norm; for earlier layers
            # we apply it too (standard logit-lens convention — it denoises).
            pass
        if final_norm is not None:
            vec = final_norm(vec.unsqueeze(0)).squeeze(0)
        logits = torch.matmul(vec, W_out.T)  # (V,)
        probs = torch.softmax(logits.float(), dim=-1)
        dom_mass = probs.index_select(0, dom_ids_t).sum().item()
        top1_id = int(torch.argmax(probs).item())
        logit_lens_rows.append({
            "layer": layer,
            "domain_token_prob_mass": float(dom_mass),
            "top1_token": tok.convert_ids_to_tokens(top1_id),
            "top1_prob": float(probs[top1_id].item()),
        })

    # attentions: tuple of n_layers tensors shape (1, n_heads, T, T).
    url_pos = _url_positions(input_ids, tok)
    if not url_pos:
        head_rows: list[dict] = []
    else:
        url_idx = torch.tensor(url_pos, device=out.attentions[0].device, dtype=torch.long)
        head_rows = []
        for layer, attn in enumerate(out.attentions):
            # attention from decision token (row=decision_pos) to URL tokens.
            a = attn[0, :, decision_pos, :]  # (n_heads, T)
            to_url = a.index_select(1, url_idx).sum(dim=1)  # (n_heads,)
            for head, v in enumerate(to_url.float().cpu().numpy()):
                head_rows.append({
                    "layer": layer,
                    "head": head,
                    "attn_to_url": float(v),
                })

    return {"logit_lens": logit_lens_rows, "heads": head_rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-n", type=int, default=100)
    ap.add_argument(
        "--model",
        default=os.getenv("PRIMARY_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        help="ORIGINAL experiment model. 70B/72B in 4-bit ≈ 42 GB VRAM.",
    )
    ap.add_argument("--proxy", action="store_true",
                    help="Use $PROXY_MODEL instead (dev only, not for paper).")
    ap.add_argument("--serp-pool", type=int, default=20)
    ap.add_argument("--serp-backend", default="searxng")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--data-root", default=None)
    ap.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "output"),
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--variant",
        choices=(
            "biased", "neutral",
            "biased_passage", "neutral_passage",
            "biased_rag", "neutral_rag",
        ),
        default=os.getenv("PROMPT_VARIANT", "biased"),
        help="Prompt variant — controls the rerank prompt that the LLM is fed "
             "(so logit-lens / attention measurements reflect THIS variant's "
             "decision context, not the default biased one). Defaults to "
             "PROMPT_VARIANT env or 'biased'.",
    )
    args = ap.parse_args()

    if args.proxy:
        args.model = os.getenv("PROXY_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        print(f"[weights] --proxy: {args.model} (DEV ONLY)")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    lens_path = out_dir / "logit_lens.csv"
    heads_path = out_dir / "attention_heads.csv"
    ckpt = Checkpoint.load(out_dir / "checkpoint_weights.json")

    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed.")
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[weights] device={device} model={args.model} variant={args.variant}")
    if device == "cpu":
        print("[weights] WARNING: CPU will be ~50x slower.")

    print(f"[weights] loading main table from {data_root(args.data_root)}")
    main_df = load_main_table(args.data_root)
    serp_df = load_serp(
        backend=args.serp_backend, pool=args.serp_pool, root=args.data_root
    )
    serp_by_kw = {k: g.sort_values("position").to_dict("records")
                  for k, g in serp_df.groupby("keyword")}
    kws = sample_keywords(main_df, args.sample_n, rng)
    print(f"[weights] keywords sampled: {len(kws)}")

    model, tok = _load_model(args.model, device)
    if device == "cuda":
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"[weights] GPU memory after load: {mem:.2f} GB")
    domain_ids = _domain_token_ids(tok)
    print(f"[weights] domain-like vocab tokens: {len(domain_ids)}")

    lens_rows: list[dict] = []
    head_rows: list[dict] = []
    header_lens = lens_path.exists() and args.resume
    header_heads = heads_path.exists() and args.resume

    for kw in tqdm(kws, desc="keywords"):
        if ckpt.seen(kw):
            continue
        cand = serp_by_kw.get(kw)
        if not cand:
            continue
        prompt = build_rerank_prompt(
            kw, cand[: args.serp_pool], top_n=args.top_n, variant=args.variant,
        )
        try:
            res = run_one(model, tok, prompt, device, args.max_len, domain_ids)
        except Exception as e:
            print(f"[weights] skip {kw}: {e}")
            continue

        for r in res["logit_lens"]:
            r["keyword"] = kw
            r["model"] = args.model
            lens_rows.append(r)
        for r in res["heads"]:
            r["keyword"] = kw
            r["model"] = args.model
            head_rows.append(r)

        ckpt.mark(kw)

        if len(lens_rows) >= 500:
            pd.DataFrame(lens_rows).to_csv(
                lens_path,
                mode=("a" if header_lens else "w"),
                header=not header_lens, index=False,
            )
            header_lens = True
            lens_rows.clear()
        if len(head_rows) >= 5000:
            pd.DataFrame(head_rows).to_csv(
                heads_path,
                mode=("a" if header_heads else "w"),
                header=not header_heads, index=False,
            )
            header_heads = True
            head_rows.clear()
        ckpt.save()

    if lens_rows:
        pd.DataFrame(lens_rows).to_csv(
            lens_path,
            mode=("a" if header_lens else "w"),
            header=not header_lens, index=False,
        )
    if head_rows:
        pd.DataFrame(head_rows).to_csv(
            heads_path,
            mode=("a" if header_heads else "w"),
            header=not header_heads, index=False,
        )
    ckpt.save()

    # Summary: top-10 attention heads by mean attn_to_url.
    if heads_path.exists():
        heads = pd.read_csv(heads_path)
        top = (heads.groupby(["layer", "head"])["attn_to_url"]
               .mean().reset_index().sort_values("attn_to_url", ascending=False)
               .head(10))
        print("[weights] top 10 (layer, head) by mean attention to URL tokens:")
        print(top.to_string(index=False))

    print(f"[weights] done -> {lens_path}, {heads_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
