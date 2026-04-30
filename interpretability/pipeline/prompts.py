"""Variant-aware rerank prompt template.

Four variants on two orthogonal axes:

  prompt-instruction: ``biased`` vs ``neutral``
  per-result content: snippet-only (default) vs ``_passage`` (passage-augmented)

- ``biased``  - byte-identical to ``pipeline/gather_data.py:_build_rerank_prompt``
                from the upstream GEODML repo. Includes the editorial exclusion
                list ("Exclude non-product sites: review aggregators, directories,
                Wikipedia, news, blogs, forums, YouTube.") and the "software
                product domains" framing. Reproduces the original DML coefficients.

- ``neutral`` - drops both the exclusion list and the "software product"
                framing. Keeps the ``Search keyword: {keyword}`` line.
                Used for the de-biased re-run that isolates prompt-instruction
                effects from the LLM's intrinsic ranking preferences.

- ``biased_passage`` / ``neutral_passage`` - same headers/footers as their
                non-passage twins, but each result is rendered as a multi-line
                block that includes ~800 chars of cleaned body text from the
                cached HTML alongside the SERP snippet. Brackets the
                snippet-only condition to test whether findings generalize to
                the richer per-source content production GEO systems feed
                their LLMs.

The active variant is selected at module import time from the ``PROMPT_VARIANT``
environment variable (default: ``biased`` for backward compatibility), or
per-call via the ``variant=`` kwarg. SLURM sbatch wrappers should set
``--export=ALL,PROMPT_VARIANT=neutral`` to switch modes for an entire job.
"""

from __future__ import annotations

import os
import re
from typing import Literal

PromptVariant = Literal["biased", "neutral", "biased_passage", "neutral_passage"]

_KNOWN_VARIANTS: tuple[str, ...] = ("biased", "neutral", "biased_passage", "neutral_passage")

# Resolved at import time. Override per-process via PROMPT_VARIANT env var or
# per-call by passing variant=.
_DEFAULT_VARIANT: PromptVariant = os.getenv("PROMPT_VARIANT", "biased")  # type: ignore[assignment]
if _DEFAULT_VARIANT not in _KNOWN_VARIANTS:
    raise ValueError(
        f"PROMPT_VARIANT must be one of {_KNOWN_VARIANTS}, got {_DEFAULT_VARIANT!r}"
    )


# Header bodies contain a {top_n} placeholder; .format() at build time.
# Passage variants reuse their non-passage twin's wording — only the per-result
# rendering differs, and the spans + downstream consumers depend on the header
# being identical between the snippet and passage runs.
_BIASED_HEADER = (
    "Below are search engine results for the above keyword. Re-rank the "
    "results and return the top {top_n} software product domains, ordered "
    "by relevance to the keyword.\n\n"
    "Exclude non-product sites: review aggregators, directories, Wikipedia, "
    "news, blogs, forums, YouTube.\n\n"
    "Return only root domains, one per line. No explanations.\n\n"
)
_NEUTRAL_HEADER = (
    "Below are search engine results for the above keyword. Re-rank the "
    "results and return the top {top_n} URLs ordered by relevance to the "
    "keyword.\n\n"
    "Return only root domains, one per line. No explanations.\n\n"
)

_HEADERS: dict[str, str] = {
    "biased":          _BIASED_HEADER,
    "neutral":         _NEUTRAL_HEADER,
    "biased_passage":  _BIASED_HEADER,
    "neutral_passage": _NEUTRAL_HEADER,
}

_FOOTERS: dict[str, str] = {
    "biased":          "\nRe-ranked product domains:",
    "neutral":         "\nRe-ranked domains:",
    "biased_passage":  "\nRe-ranked product domains:",
    "neutral_passage": "\nRe-ranked domains:",
}

_PASSAGE_MAX_CHARS = 800


def _extract_domain(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url)
    host = m.group(1) if m else url
    if host.startswith("www."):
        host = host[4:]
    return host


def _resolve(variant: PromptVariant | None) -> PromptVariant:
    return variant if variant is not None else _DEFAULT_VARIANT


def active_variant() -> PromptVariant:
    """Return the currently active default variant (read at module import)."""
    return _DEFAULT_VARIANT


def is_passage_variant(variant: PromptVariant | None = None) -> bool:
    """True iff the variant injects per-result body-text passages."""
    return _resolve(variant).endswith("_passage")


def build_rerank_prompt_with_spans(
    keyword: str,
    search_results: list[dict],
    top_n: int = 10,
    variant: PromptVariant | None = None,
) -> tuple[str, list[dict]]:
    """Build the rerank prompt and per-result character spans.

    Returns:
        prompt: the full prompt string
        spans: one dict per result with keys
            - position, url, domain
            - line_span: (char_start, char_end) of the per-result block within
              the prompt. For snippet variants this is one line; for passage
              variants it spans the full multi-line block (so probing's
              per-result hidden-state pool covers the body content too).
            - domain_span: (char_start, char_end) of the bare domain (no
              brackets) within prompt - this is the only T7 signal the LLM
              actually sees, regardless of variant.

    With ``variant="biased"`` (the default for backward compatibility) the
    output is byte-identical to the original
    ``interpretability.utils.build_rerank_prompt_with_spans`` (pre-port) and
    to ``pipeline/gather_data.py:_build_rerank_prompt`` upstream.
    """
    v = _resolve(variant)
    passage_mode = v.endswith("_passage")
    header_top = (
        f"Search keyword: {keyword}\n\n"
        + _HEADERS[v].format(top_n=top_n)
        + "Search results:\n"
    )

    spans: list[dict] = []
    parts: list[str] = []
    cursor = 0  # offset within the assembled results_text
    for r in search_results:
        domain = _extract_domain(r["url"])
        snippet = (r.get("snippet") or "")[:150]
        prefix = f"{r['position']}. ["

        if passage_mode:
            passage = (r.get("passage") or "")[:_PASSAGE_MAX_CHARS]
            line = (
                f"{prefix}{domain}] {r['title']}\n"
                f"   snippet: {snippet}\n"
                f"   passage: {passage}\n"
            )
        else:
            line = f"{prefix}{domain}] {r['title']} — {snippet}\n"

        line_start = cursor
        domain_start = cursor + len(prefix)
        domain_end = domain_start + len(domain)
        line_end = cursor + len(line)
        parts.append(line)
        spans.append({
            "position": r["position"],
            "url": r["url"],
            "domain": domain,
            "line_span": (line_start, line_end),
            "domain_span": (domain_start, domain_end),
        })
        cursor = line_end
    results_text = "".join(parts)

    prompt = header_top + results_text + _FOOTERS[v]
    base = len(header_top)
    for s in spans:
        ls, le = s["line_span"]
        ds, de = s["domain_span"]
        s["line_span"] = (base + ls, base + le)
        s["domain_span"] = (base + ds, base + de)

    return prompt, spans


def build_rerank_prompt(
    keyword: str,
    search_results: list[dict],
    top_n: int = 10,
    variant: PromptVariant | None = None,
) -> str:
    """Build the rerank prompt string (no spans). See ``build_rerank_prompt_with_spans``."""
    prompt, _ = build_rerank_prompt_with_spans(
        keyword, search_results, top_n=top_n, variant=variant,
    )
    return prompt
