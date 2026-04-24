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


def build_rerank_prompt(
    keyword: str, search_results: list[dict], top_n: int = 10
) -> str:
    """Exact template used in the original experiment.

    `search_results` items must have keys: position, title, url, snippet.
    """
    results_text = ""
    for r in search_results:
        domain = _extract_domain(r["url"])
        snippet = (r.get("snippet") or "")[:150]
        results_text += (
            f"{r['position']}. [{domain}] {r['title']} — {snippet}\n"
        )
    return (
        f"Search keyword: {keyword}\n\n"
        f"Below are search engine results for the above keyword. Re-rank the "
        f"results and return the top {top_n} software product domains, ordered "
        f"by relevance to the keyword.\n\n"
        f"Exclude non-product sites: review aggregators, directories, Wikipedia, "
        f"news, blogs, forums, YouTube.\n\n"
        f"Return only root domains, one per line. No explanations.\n\n"
        f"Search results:\n{results_text}\n"
        f"Re-ranked product domains:"
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
# HF Inference API client (for ablation.py)
# ---------------------------------------------------------------------------

class InferenceRanker:
    """Thin wrapper over huggingface_hub.InferenceClient.chat_completion."""

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
URL_HINT_RE = re.compile(r"(https?://|www\.|\.com|\.io|\.org|\.net)", re.I)
EARNED_HINT_RE = re.compile(r"\b(official|review|best|top|vs|alternatives?)\b", re.I)


def tag_token_for_treatment(tok: str) -> list[str]:
    """Return all treatment tags that this token matches."""
    tags: list[str] = []
    if NUMERIC_RE.search(tok):
        tags.append("T1b_stats_density")
    if QUESTION_RE.search(tok):
        tags.append("T2a_question_headings")
    if SCHEMA_RE.search(tok) or YEAR_RE.search(tok):
        if SCHEMA_RE.search(tok):
            tags.append("T3_structured_data_new")
        if YEAR_RE.search(tok):
            tags.append("T6_freshness")
    if URL_HINT_RE.search(tok) or EARNED_HINT_RE.search(tok):
        tags.append("T7_source_earned")
    return tags
