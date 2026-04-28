"""Pipeline configuration: model list, pool sizes, treatments, DML grid.

Ported (and trimmed) from ``paperSizeExperiment/config.py`` upstream.

What was DROPPED in the port:

- All HTTP-touched API keys (SEARXNG_URL, OPENPAGERANK_KEY, MOZ_API_KEY,
  KAGI_TOKEN, BRAVE_API_KEY, GOOGLE_API_KEY/CX, SERPAPI_KEY) - the new
  pipeline never hits the live web. SERPs and HTML are reused from the cached
  HF dataset; nothing leaves the cluster.
- ``ENABLE_LLM_FEATURES`` defaulted False (deterministic-only T1-T4 by decision).
- ``ENABLE_PAGERANK``, ``ENABLE_WHOIS`` dropped (HTTP).
- ``FETCH_TIMEOUT``, ``MAX_HTML_SIZE``, ``USER_AGENT`` dropped (no fetch).
- ``TREATMENTS_LLM`` mapping kept for backward compatibility with legacy DML
  tables, but the pipeline does not write those columns by default. To enable,
  set ``ENABLE_LLM_TREATMENTS=1`` in the env (off by default).

The variant selection (``biased`` vs ``neutral``) lives in
``interpretability.pipeline.prompts`` and is configured via ``PROMPT_VARIANT``.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Experiment grid ───────────────────────────────────────────────────────────

# LLM models for re-ranking (HuggingFace IDs).
LLM_MODELS: list[str] = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
]

# Pool sizes: (serp_results, llm_top_n).
POOL_SIZES: list[tuple[int, int]] = [
    (20, 10),  # small pool: 20 SERP results -> top 10
    (50, 10),  # large pool: 50 SERP results -> top 10
]

# Cached engines available in the HF dataset under data/serp/phase0_*.parquet.
ENGINES: list[str] = ["searxng", "ddg"]

# Map ENGINES element -> phase0 parquet basename suffix (e.g. "searxng" -> "searxng",
# "ddg" -> "ddg"). Kept explicit because the upstream used "duckduckgo" sometimes.
ENGINE_TO_PHASE0_SUFFIX: dict[str, str] = {
    "searxng": "searxng",
    "ddg": "ddg",
}

# LLM rerank generation params (matching original).
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 500

# ── Pipeline toggles ──────────────────────────────────────────────────────────

# Optional LLM-based treatment scoring (T1_llm..T4_llm). Off by default — pure
# regex/BS4/JSON-LD parsing for T1a/T1b/T2a/T2b/T3/T4a/T4b/T6/T7 is the
# canonical path post-port. Set ENABLE_LLM_TREATMENTS=1 to also produce the
# legacy LLM-scored columns as a robustness check.
ENABLE_LLM_TREATMENTS: bool = bool(int(os.getenv("ENABLE_LLM_TREATMENTS", "0")))

# ── DML analysis settings ─────────────────────────────────────────────────────

DML_METHODS: list[str] = ["plr"]            # ["plr", "irm"]
DML_LEARNERS: list[str] = ["lgbm", "rf"]    # sensitivity check
DML_N_FOLDS: int = 5
DML_OUTCOMES: list[str] = ["rank_delta", "post_rank"]

# ── Treatment definitions ─────────────────────────────────────────────────────
# Maps short label -> column name in the merged dataset.

TREATMENTS_CODE: dict[str, str] = {
    "T1_code": "T1_statistical_density_code",
    "T2_code": "T2_question_heading_code",
    "T3_code": "T3_structured_data_code",
    "T4_code": "T4_citation_authority_code",
}

TREATMENTS_LLM: dict[str, str] = {
    "T1_llm": "T1_statistical_density_llm",
    "T2_llm": "T2_question_heading_llm",
    "T3_llm": "T3_structured_data_llm",
    "T4_llm": "T4_citation_authority_llm",
}

TREATMENTS_NEW: dict[str, str] = {
    "T1a_stats_present":         "treat_stats_present",
    "T1b_stats_density":         "treat_stats_density",
    "T2a_question_headings":     "treat_question_headings",
    "T2b_structural_modularity": "treat_structural_modularity",
    "T3_structured_data_new":    "treat_structured_data",
    "T4a_ext_citations":         "treat_ext_citations_any",
    "T4b_auth_citations":        "treat_auth_citations",
    "T5_topical_comp":           "treat_topical_comp",
    "T6_freshness":              "treat_freshness",
    "T7_source_earned":          "treat_source_earned",
}

ALL_TREATMENTS: dict[str, str] = {
    **TREATMENTS_CODE, **TREATMENTS_LLM, **TREATMENTS_NEW,
}

TREATMENT_LABELS: dict[str, str] = {
    "T1_code":                   "T1 Statistical Density (code)",
    "T2_code":                   "T2 Question Headings (code)",
    "T3_code":                   "T3 Structured Data (code)",
    "T4_code":                   "T4 Citation Authority (code)",
    "T1_llm":                    "T1 Statistical Density (LLM)",
    "T2_llm":                    "T2 Question Headings (LLM)",
    "T3_llm":                    "T3 Structured Data (LLM)",
    "T4_llm":                    "T4 Citation Authority (LLM)",
    "T1a_stats_present":         "T1a Stats Present (binary)",
    "T1b_stats_density":         "T1b Stats Density (continuous)",
    "T2a_question_headings":     "T2a Question Headings (binary)",
    "T2b_structural_modularity": "T2b Structural Modularity (count)",
    "T3_structured_data_new":    "T3 Structured Data (expanded)",
    "T4a_ext_citations":         "T4a External Citations (binary)",
    "T4b_auth_citations":        "T4b Authority Citations (count)",
    "T5_topical_comp":           "T5 Topical Competence (cosine)",
    "T6_freshness":              "T6 Freshness (ordinal 0-4)",
    "T7_source_earned":          "T7 Source: Earned",
}

# ── Confounders ───────────────────────────────────────────────────────────────
# The full set used by the most recent DML run. Items prefixed with "dfs_" are
# DataForSEO keyword-level metrics that ARE present in the cached parquet
# (joined upstream); the new pipeline does not refetch them. "conf_domain_authority",
# "conf_backlinks", and "conf_referring_domains" are Moz-derived columns; if the
# Moz cache parquet is not available, these columns will be NaN and DML excludes
# them automatically.

CONFOUNDERS: list[str] = [
    "conf_title_kw_sim",
    "conf_snippet_kw_sim",
    "conf_title_len",
    "conf_snippet_len",
    "conf_brand_recog",
    "conf_title_has_kw",
    "conf_word_count",
    "conf_readability",
    "conf_internal_links",
    "conf_outbound_links",
    "conf_images_alt",
    "conf_bm25",
    "conf_https",
    "conf_domain_authority",
    "conf_backlinks",
    "conf_referring_domains",
    "conf_serp_position",
    # DataForSEO keyword-level confounders (joined from cached parquet)
    "dfs_keyword_difficulty",
    "dfs_search_volume",
    "dfs_cpc",
    "dfs_competition",
    "dfs_intent_commercial",
    "dfs_intent_informational",
    "dfs_intent_navigational",
    "dfs_intent_transactional",
]

# Legacy confounder set, kept for reproduction of the original biased numbers.
CONFOUNDERS_LEGACY: list[str] = [
    "X1_domain_authority",
    "X2_domain_age_years",
    "X3_word_count",
    "X6_readability",
    "X7_internal_links",
    "X7B_outbound_links",
    "X8_keyword_difficulty",
    "X9_images_with_alt",
]


# ── Run-id helpers ────────────────────────────────────────────────────────────

def run_label(engine: str, model_id: str, serp_n: int, llm_top_n: int) -> str:
    """Short label for an experiment run, matches upstream layout.

    Example: ``searxng_Llama-3.3-70B-Instruct_serp50_top10``.
    """
    model_short = model_id.split("/")[-1]
    return f"{engine}_{model_short}_serp{serp_n}_top{llm_top_n}"


def run_label_with_variant(engine: str, model_id: str, serp_n: int,
                           llm_top_n: int, variant: str) -> str:
    """Run label suffixed with the prompt variant.

    The neutral and biased variants must coexist on disk for side-by-side
    comparison, so each variant gets its own run dir.
    """
    if variant not in ("biased", "neutral"):
        raise ValueError(f"variant must be 'biased' or 'neutral', got {variant!r}")
    return f"{run_label(engine, model_id, serp_n, llm_top_n)}_{variant}"


def short_model_name(model_id: str) -> str:
    """``meta-llama/Llama-3.3-70B-Instruct`` -> ``Llama-3.3-70B-Instruct``."""
    return model_id.split("/")[-1]


# ── Output paths (relative to GEODML_Analysis repo root) ──────────────────────

DEFAULT_DATA_ROOT = Path(os.getenv("GEODML_DATA_ROOT", "geodml_data")).resolve()


def runs_dir(data_root: Path | None = None) -> Path:
    return (data_root or DEFAULT_DATA_ROOT) / "data" / "runs"


def serp_path(engine: str, pool: int, data_root: Path | None = None) -> Path:
    suf = ENGINE_TO_PHASE0_SUFFIX[engine]
    return (data_root or DEFAULT_DATA_ROOT) / "data" / "serp" / f"phase0_top{pool}_{suf}.parquet"


def main_table_path(variant: str, data_root: Path | None = None) -> Path:
    """The merged DML-ready parquet for a given variant."""
    return (data_root or DEFAULT_DATA_ROOT) / "data" / "main" / f"full_experiment_data_{variant}.parquet"


def dml_results_path(variant: str, data_root: Path | None = None) -> Path:
    return (data_root or DEFAULT_DATA_ROOT) / "data" / "dml_results" / f"dml_results_long_{variant}.parquet"
