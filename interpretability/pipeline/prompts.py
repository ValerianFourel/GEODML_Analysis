"""Variant-aware rerank prompt template.

Two variants:

- ``biased``  - byte-identical to ``pipeline/gather_data.py:_build_rerank_prompt``
                from the upstream GEODML repo. Includes the editorial exclusion
                list ("Exclude non-product sites: review aggregators, directories,
                Wikipedia, news, blogs, forums, YouTube.") and the "software
                product domains" framing. Reproduces the original DML coefficients.

- ``neutral`` - drops both the exclusion list and the "software product"
                framing. Keeps the ``Search keyword: {keyword}`` line.
                Used for the de-biased re-run that isolates prompt-instruction
                effects from the LLM's intrinsic ranking preferences.

The active variant is selected at module import time from the ``PROMPT_VARIANT``
environment variable (default: ``biased`` for backward compatibility), or
per-call via the ``variant=`` kwarg. SLURM sbatch wrappers should set
``--export=ALL,PROMPT_VARIANT=neutral`` to switch modes for an entire job.
"""

from __future__ import annotations

import os
import re
from typing import Literal

PromptVariant = Literal["biased", "neutral"]

# Resolved at import time. Override per-process via PROMPT_VARIANT env var or
# per-call by passing variant=.
_DEFAULT_VARIANT: PromptVariant = os.getenv("PROMPT_VARIANT", "biased")  # type: ignore[assignment]
if _DEFAULT_VARIANT not in ("biased", "neutral"):
    raise ValueError(
        f"PROMPT_VARIANT must be 'biased' or 'neutral', got {_DEFAULT_VARIANT!r}"
    )


# Header bodies contain a {top_n} placeholder; .format() at build time.
_HEADERS: dict[str, str] = {
    "biased": (
        "Below are search engine results for the above keyword. Re-rank the "
        "results and return the top {top_n} software product domains, ordered "
        "by relevance to the keyword.\n\n"
        "Exclude non-product sites: review aggregators, directories, Wikipedia, "
        "news, blogs, forums, YouTube.\n\n"
        "Return only root domains, one per line. No explanations.\n\n"
    ),
    "neutral": (
        "Below are search engine results for the above keyword. Re-rank the "
        "results and return the top {top_n} URLs ordered by relevance to the "
        "keyword.\n\n"
        "Return only root domains, one per line. No explanations.\n\n"
    ),
}

_FOOTERS: dict[str, str] = {
    "biased":  "\nRe-ranked product domains:",
    "neutral": "\nRe-ranked domains:",
}


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
            - line_span: (char_start, char_end) of the result line within prompt
            - domain_span: (char_start, char_end) of the bare domain (no brackets)
              within prompt - this is the only T7 signal the LLM actually sees.

    With ``variant="biased"`` (the default for backward compatibility) the
    output is byte-identical to the original
    ``interpretability.utils.build_rerank_prompt_with_spans`` (pre-port) and
    to ``pipeline/gather_data.py:_build_rerank_prompt`` upstream.
    """
    v = _resolve(variant)
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
