"""Shared utilities: data loader, HTML loader, prompt builder, HF client, checkpointer."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUN_IDS = [
    "duckduckgo_Llama-3.3-70B-Instruct_serp20_top10",
    "duckduckgo_Llama-3.3-70B-Instruct_serp50_top10",
    "duckduckgo_Qwen2.5-72B-Instruct_serp20_top10",
    "duckduckgo_Qwen2.5-72B-Instruct_serp50_top10",
    "searxng_Llama-3.3-70B-Instruct_serp20_top10",
    "searxng_Llama-3.3-70B-Instruct_serp50_top10",
    "searxng_Qwen2.5-72B-Instruct_serp20_top10",
    "searxng_Qwen2.5-72B-Instruct_serp50_top10",
]

# Mapping from DML treatment labels to columns in full_experiment_data.parquet.
TREATMENT_TO_COL = {
    "T1_code": "T1_statistical_density_code",
    "T1_llm": "T1_statistical_density_llm",
    "T1b_stats_density": "treat_stats_density",
    "T2_code": "T2_question_heading_code",
    "T2_llm": "T2_question_heading_llm",
    "T2a_question_headings": "treat_question_headings",
    "T2b_structural_modularity": "treat_structural_modularity",
    "T3_code": "T3_structured_data_code",
    "T3_llm": "T3_structured_data_llm",
    "T3_structured_data_new": "treat_structured_data",
    "T4_code": "T4_citation_authority_code",
    "T4_llm": "T4_citation_authority_llm",
    "T4b_auth_citations": "treat_auth_citations",
    "T5_topical_comp": "treat_topical_comp",
    "T6_freshness": "treat_freshness",
    "T7_source_earned": "treat_source_earned",
    "T_llms_txt": "has_llms_txt",
}


def data_root(explicit: str | Path | None = None) -> Path:
    root = Path(explicit or os.getenv("GEODML_DATA_ROOT", "./geodml_data"))
    return root.resolve()


def hf_token() -> str:
    t = os.getenv("HF_TOKEN")
    if not t:
        raise RuntimeError("HF_TOKEN not set. Copy .env.example to .env and fill it in.")
    return t


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_main_table(root: str | Path | None = None) -> pd.DataFrame:
    """Primary 65k-row table with treatments, confounders, outcomes."""
    r = data_root(root)
    return pd.read_parquet(r / "data" / "main" / "full_experiment_data.parquet")


def load_dml_long(root: str | Path | None = None) -> pd.DataFrame:
    """570 DML fits (coef, SE, p_val, stars, subset, outcome)."""
    r = data_root(root)
    return pd.read_parquet(r / "data" / "dml_results" / "dml_results_long.parquet")


def load_serp(
    backend: str = "searxng", pool: int = 50, root: str | Path | None = None
) -> pd.DataFrame:
    """Phase-0 SERP snapshot. columns: keyword, position, title, url, snippet, engines, score."""
    if backend not in ("searxng", "ddg"):
        raise ValueError(f"backend must be searxng or ddg, got {backend}")
    if pool not in (20, 50):
        raise ValueError(f"pool must be 20 or 50, got {pool}")
    r = data_root(root)
    return pd.read_parquet(r / "data" / "serp" / f"phase0_top{pool}_{backend}.parquet")


def load_run_features(run_id: str, root: str | Path | None = None) -> pd.DataFrame:
    """Per-run features.parquet — keyed by url, includes per-URL features + error state."""
    r = data_root(root)
    return pd.read_parquet(r / "data" / "runs" / run_id / "phase2" / "features.parquet")


def load_run_keywords(run_id: str, root: str | Path | None = None) -> Iterable[dict]:
    """Yield per-keyword JSONL records from phase2/keywords.jsonl (streaming)."""
    r = data_root(root)
    p = r / "data" / "runs" / run_id / "phase2" / "keywords.jsonl"
    with open(p) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ---------------------------------------------------------------------------
# HTML loader (handles both extracted cache and tar.gz form transparently)
# ---------------------------------------------------------------------------

def url_to_html_filename(url: str) -> str:
    """Convention used by the original pipeline: md5(url)[:16] + '.html'."""
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:16] + ".html"


class HTMLLoader:
    """Reads per-run page HTML.

    Prefers the extracted `html_cache/` directory if present; falls back to
    streaming from `html_cache.tar.gz` (slower, but no disk blowup needed).
    """

    def __init__(self, run_id: str, root: str | Path | None = None):
        self.run_id = run_id
        self.root = data_root(root) / "data" / "runs" / run_id / "phase2"
        self.extracted = self.root / "html_cache"
        self.tarball = self.root / "html_cache.tar.gz"
        self._tar: tarfile.TarFile | None = None
        self._tar_names: set[str] | None = None

    def _open_tar(self) -> tarfile.TarFile:
        if self._tar is None:
            if not self.tarball.exists():
                raise FileNotFoundError(
                    f"Neither {self.extracted} nor {self.tarball} exists for run {self.run_id}"
                )
            self._tar = tarfile.open(self.tarball, "r:gz")
            self._tar_names = {m.name for m in self._tar.getmembers()}
        return self._tar

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def get_html(self, url: str) -> str | None:
        fname = url_to_html_filename(url)
        if self.extracted.exists():
            p = self.extracted / fname
            if p.exists():
                return p.read_text(encoding="utf-8", errors="replace")
            return None
        tar = self._open_tar()
        candidates = [f"html_cache/{fname}", fname]
        for c in candidates:
            if c in (self._tar_names or set()):
                m = tar.extractfile(c)
                if m is None:
                    return None
                return m.read().decode("utf-8", errors="replace")
        return None


def page_digest(html: str, max_chars: int = 8000) -> str:
    """Strip HTML to readable body text. Keep paragraphs and headings."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        tag.decompose()
    parts: list[str] = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        txt = el.get_text(" ", strip=True)
        if txt:
            parts.append(txt)
    text = "\n".join(parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Re-ranking prompt (reverse-engineered from src/llm_ranker.py::_build_prompt)
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url)
    host = m.group(1) if m else url
    if host.startswith("www."):
        host = host[4:]
    return host


# Re-exported from interpretability.pipeline.prompts so the variant selection
# (PROMPT_VARIANT env var / variant= kwarg) is honored in every call site
# (ablation.py, saliency.py, probing.py, weight_analysis.py).
# The biased default reproduces the original prompt byte-for-byte.
from interpretability.pipeline.prompts import (  # noqa: E402
    build_rerank_prompt,
    build_rerank_prompt_with_spans,
)


def parse_ranked_domains(llm_output: str) -> list[str]:
    """Parse newline-separated, possibly-numbered domain list."""
    out: list[str] = []
    llm_output = re.sub(r"<think>.*?</think>", "", llm_output, flags=re.DOTALL).strip()
    for line in llm_output.splitlines():
        line = line.strip()
        if not line:
            continue
        for prefix in ("- ", "* "):
            if line.startswith(prefix):
                line = line[len(prefix):]
        if line and line[0].isdigit():
            # strip "1." or "1)" style numbering
            line = re.sub(r"^\s*\d+[\.\)]\s*", "", line)
        line = line.strip().strip("`").strip()
        if line:
            out.append(line.lower())
    return out


# ---------------------------------------------------------------------------
# Multi-GPU sharding helpers (Jülich booster: --gres=gpu:4 on H100/A100)
# ---------------------------------------------------------------------------

def multi_gpu_load_kwargs(
    quantize: bool = True,
    reserve_gib_per_gpu: float = 8.0,
    cpu_offload_gib: int = 64,
    output_hidden_states: bool = False,
) -> dict:
    """Build `from_pretrained` kwargs that shard a model across visible GPUs.

    Sets `device_map="auto"` plus a `max_memory` budget that reserves
    `reserve_gib_per_gpu` on each device for activations / KV cache / grads.
    On 4-GPU SLURM jobs (`--gres=gpu:4`), this forces accelerate to spread a
    70B-bf16 (~140 GB) or a saliency forward+backward across multiple cards
    instead of OOMing on cuda:0.

    Single-GPU and CPU paths are unchanged.
    """
    import torch

    kw: dict = {}
    if output_hidden_states:
        kw["output_hidden_states"] = True

    if not torch.cuda.is_available():
        kw["torch_dtype"] = torch.float32
        return kw

    n_gpus = torch.cuda.device_count()
    kw["device_map"] = "auto"

    if n_gpus > 1:
        budgets: dict = {}
        for i in range(n_gpus):
            total_gib = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            usable = max(int(total_gib - reserve_gib_per_gpu), 1)
            budgets[i] = f"{usable}GiB"
        budgets["cpu"] = f"{cpu_offload_gib}GiB"
        kw["max_memory"] = budgets
        print(
            f"[load] CUDA_VISIBLE_DEVICES gives {n_gpus} GPUs; "
            f"max_memory={ {k: v for k, v in budgets.items() if k != 'cpu'} } "
            f"(reserved {reserve_gib_per_gpu:.0f} GiB/GPU for activations/grads)"
        )

    if quantize:
        try:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except Exception as e:
            print(f"[load] bitsandbytes unavailable ({e}); falling back to fp16")
            kw["torch_dtype"] = torch.float16
    return kw


def log_device_map(model, prefix: str = "[load]") -> None:
    """Compact SLURM-log-friendly print of how the model was sharded."""
    dm = getattr(model, "hf_device_map", None)
    if not dm:
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = "?"
        print(f"{prefix} single-device load: {dev}")
        return
    counts: dict = {}
    for _, dev in dm.items():
        key = str(dev)
        counts[key] = counts.get(key, 0) + 1
    summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    print(f"{prefix} hf_device_map: {len(dm)} modules over {len(counts)} devices [{summary}]")
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_allocated(i) / (1024**3)
            print(f"{prefix}   cuda:{i} allocated={mem:.2f} GiB")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Ranker backends (API for online, Local for offline HPC clusters)
# ---------------------------------------------------------------------------

class InferenceRanker:
    """Online: thin wrapper over huggingface_hub.InferenceClient.chat_completion."""

    def __init__(self, model: str, token: str | None = None, max_retries: int = 4):
        from huggingface_hub import InferenceClient

        self.model = model
        self.client = InferenceClient(token=token or hf_token())
        self.max_retries = max_retries

    def rank(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content
            except Exception as e:
                last_err = e
                sleep = min(60, 2 ** attempt)
                time.sleep(sleep)
        raise RuntimeError(f"HF inference failed after {self.max_retries} retries: {last_err}")


class LocalRanker:
    """Offline: load a HF causal-LM locally (4-bit if CUDA is available) and
    generate the re-ranking output. Use this on air-gapped clusters like Jülich.

    Handles both chat-template-capable models (Llama-3.*-Instruct, Qwen-Instruct)
    and raw-prompt fallback.
    """

    def __init__(self, model: str, quantize: bool = True, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model
        self.tok = AutoTokenizer.from_pretrained(model, use_fast=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        kw = multi_gpu_load_kwargs(quantize=quantize)
        if not torch.cuda.is_available():
            kw["torch_dtype"] = getattr(torch, dtype, torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw)
        self.model.eval()
        log_device_map(self.model, prefix="[LocalRanker]")
        self.device = next(self.model.parameters()).device
        self._has_chat_template = bool(getattr(self.tok, "chat_template", None))

    def rank(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        import torch

        attention_mask = None
        if self._has_chat_template:
            messages = [{"role": "user", "content": prompt}]
            tok_out = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            # Some transformers versions (>= 4.45) return a BatchEncoding here
            # instead of a raw Tensor; passing the dict to model.generate then
            # blows up with `BatchEncoding has no attribute 'shape'`. Normalise.
            if hasattr(tok_out, "input_ids"):
                input_ids = tok_out["input_ids"]
                attention_mask = tok_out.get("attention_mask")
            else:
                input_ids = tok_out
        else:
            enc = self.tok(prompt, return_tensors="pt")
            input_ids = enc.input_ids
            attention_mask = enc.get("attention_mask")

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=1.0,
                pad_token_id=self.tok.eos_token_id,
            )
        # Strip the prompt tokens — only return the generated continuation.
        gen = out[0, input_ids.shape[-1]:]
        return self.tok.decode(gen, skip_special_tokens=True)


def make_ranker(backend: str, model: str) -> "InferenceRanker | LocalRanker":
    """Factory. backend ∈ {'api', 'local'}."""
    if backend == "api":
        return InferenceRanker(model=model)
    if backend == "local":
        return LocalRanker(model=model)
    raise ValueError(f"unknown backend: {backend}")


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    path: Path
    data: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Checkpoint":
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            try:
                data = json.loads(p.read_text())
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}
        return cls(path=p, data=data)

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str))
        tmp.replace(self.path)

    def seen(self, key: str) -> bool:
        return key in self.data.get("seen", [])

    def mark(self, key: str) -> None:
        self.data.setdefault("seen", [])
        if key not in self.data["seen"]:
            self.data["seen"].append(key)

    def set(self, k: str, v: Any) -> None:
        self.data[k] = v


# ---------------------------------------------------------------------------
# Ablation rules (shared between ablation.py + saliency token tagging)
# ---------------------------------------------------------------------------

ABLATION_RULES = {
    # Treatment that we want to remove ("ablate") from the snippet/digest.
    # Each rule is (regex, replacement). Applied to each search result's snippet.
    "T1b_stats_density": (re.compile(r"\b\d[\d.,%$]*\b"), ""),
    "T2a_question_headings": (re.compile(r"[?]"), ""),
    "T3_structured_data_new": (
        re.compile(r"\b(schema\.org|json-ld|structured data|microdata)\b", re.I),
        "",
    ),
    "T6_freshness": (
        re.compile(
            r"\b(\d{1,2}\s+(days?|weeks?|months?|years?)\s+ago"
            r"|\d{4}-\d{2}-\d{2}"
            r"|(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"
            r"|Q[1-4]\s*\d{4})\b",
            re.I,
        ),
        "",
    ),
}


def ablate_snippet(snippet: str, treatment: str) -> str:
    """Apply treatment-specific ablation to a snippet string."""
    rule = ABLATION_RULES.get(treatment)
    if not rule:
        return snippet
    pat, rep = rule
    out = pat.sub(rep, snippet or "")
    return re.sub(r"\s+", " ", out).strip()


def ablate_t7(title: str) -> str:
    """T7 ablation for earned-media pages: prepend 'Official vendor page:'
    to simulate the brand-source signal."""
    return f"Official vendor page: {title}".strip()


# ---------------------------------------------------------------------------
# Token tagging (for saliency aggregation)
# ---------------------------------------------------------------------------

NUMERIC_RE = re.compile(r"\b\d+\b")
QUESTION_RE = re.compile(r"\?")
SCHEMA_RE = re.compile(r"(schema|structured|json-?ld)", re.I)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def tag_token_for_treatment(tok: str) -> list[str]:
    """Return all treatment tags that this token matches.

    Note: T7_source_earned is intentionally NOT tagged here. T7 is a
    domain-list classification — the LLM's only signal is the bracketed
    [domain] in each result line, conditional on that URL being earned
    media. Saliency tags T7 from prompt-level character spans instead
    (see build_rerank_prompt_with_spans + saliency.py).
    """
    tags: list[str] = []
    if NUMERIC_RE.search(tok):
        tags.append("T1b_stats_density")
    if QUESTION_RE.search(tok):
        tags.append("T2a_question_headings")
    if SCHEMA_RE.search(tok):
        tags.append("T3_structured_data_new")
    if YEAR_RE.search(tok):
        tags.append("T6_freshness")
    return tags
