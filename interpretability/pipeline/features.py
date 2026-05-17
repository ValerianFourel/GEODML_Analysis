"""Stage B - deterministic feature extraction from cached HTML.

For every (keyword, url) pair in a phase0 SERP, read the cached HTML, parse
it, and emit one row of treatments + confounders.

By design this stage is **variant-agnostic**: the page-content features do
not depend on which prompt the LLM saw. So features are extracted once per
(engine, pool) and joined onto each rerank variant by ``merge.py``.

Inputs (no HTTP):
    geodml_data/data/serp/phase0_top{20,50}_{searxng,ddg}.parquet
    geodml_data/data/runs/<any run_id with this engine+pool>/phase2/html_cache.tar.gz

Outputs:
    geodml_data/data/features/features_{engine}_top{pool}.parquet
    geodml_data/data/features/.features_{engine}_top{pool}_ckpt.json

Ported (deterministic-only) from ``pipeline/extract_features.py``:

    extract_t1a_stats_present, extract_t1b_stats_density,
    extract_t2a_question_headings, extract_t2b_structural_modularity,
    extract_t3_structured_data, extract_t4a_ext_citations_any,
    extract_t4b_auth_citations, extract_t6_freshness, classify_source_type,
    conf_title_has_kw, conf_brand_recog, extract_word_count,
    extract_readability, extract_internal_links, extract_outbound_links,
    extract_images_alt, compute_embeddings, cosine_sim, compute_bm25_scores

Dropped (HTTP / superseded):
    fetch_moz_data, _load_html, _url_to_cache_key, _get_soup,
    _extract_body_text, _extract_domain (use utils versions),
    llm_extract_treatments (deterministic-only by decision).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from copy import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, Comment
from tqdm import tqdm

from interpretability.pipeline import config as C
from interpretability.utils import (
    Checkpoint,
    HTMLLoader,
    _extract_domain,
    data_root,
    load_serp,
)


# ── Static domain sets (verbatim from extract_features.py) ───────────────────

BRAND_DOMAINS: set[str] = {
    "salesforce.com", "hubspot.com", "microsoft.com", "oracle.com",
    "sap.com", "adobe.com", "google.com", "ibm.com", "cisco.com",
    "servicenow.com", "workday.com", "zendesk.com", "atlassian.com",
    "slack.com", "zoom.us", "dropbox.com", "shopify.com", "twilio.com",
    "datadog.com", "snowflake.com", "cloudflare.com", "okta.com",
    "pagerduty.com", "elastic.co", "mongodb.com", "confluent.io",
    "hashicorp.com", "databricks.com", "stripe.com", "brevo.com",
    "mailchimp.com", "intercom.com", "freshworks.com", "zoho.com",
    "monday.com", "asana.com", "notion.so", "airtable.com",
    "clickup.com", "smartsheet.com", "wix.com", "squarespace.com",
    "bigcommerce.com", "klaviyo.com", "semrush.com", "ahrefs.com",
    "moz.com", "hootsuite.com", "buffer.com", "sproutsocial.com",
    "canva.com", "figma.com", "webflow.com", "unbounce.com",
    "activecampaign.com", "drift.com", "gong.io", "outreach.io",
    "salesloft.com", "docusign.com", "pandadoc.com", "calendly.com",
    "loom.com", "vidyard.com", "wistia.com", "typeform.com",
    "surveymonkey.com", "qualtrics.com", "amplitude.com", "mixpanel.com",
    "segment.com", "braze.com", "iterable.com", "appcues.com",
    "pendo.io", "gainsight.com", "totango.com", "churnzero.com",
    "chargebee.com", "recurly.com", "zuora.com", "bill.com",
    "xero.com", "quickbooks.intuit.com", "netsuite.com", "sage.com",
    "gusto.com", "rippling.com", "bamboohr.com", "paylocity.com",
    "paycom.com", "deel.com", "remote.com", "oysterhr.com",
    "lever.co", "greenhouse.io", "ashbyhq.com", "gem.com",
    "procore.com", "autodesk.com", "plangrid.com", "buildertrend.com",
    "toast.com", "lightspeed.com", "mindbodyonline.com",
    "zenoti.com", "veeva.com", "clio.com", "appfolio.com",
}

EARNED_DOMAINS: set[str] = {
    "g2.com", "capterra.com", "trustradius.com", "softwareadvice.com",
    "getapp.com", "gartner.com", "forrester.com", "idc.com",
    "solutionsreview.com", "selecthub.com", "betterbuys.com",
    "peerspot.com", "sourceforge.net", "crozdesk.com", "financesonline.com",
    "goodfirms.co", "trustpilot.com", "alternativeto.net", "softwaresuggest.com",
    "technologyadvice.com", "saashub.com", "clutch.co", "stackshare.io",
    "featuredcustomers.com", "saasworthy.com", "betalist.com", "indiehackers.com",
    "serchen.com", "saasgenius.com", "crowdreviews.com", "f6s.com",
    "startupstash.com", "saasmag.com", "softwarereviews.com", "spiceworks.com",
    "infotech.com", "toolradar.com", "selectsoftwarereviews.com",
    "discovercrm.com", "emailvendorselection.com",
    "techcrunch.com", "venturebeat.com", "zdnet.com", "techradar.com",
    "pcmag.com", "cnet.com", "tomsguide.com", "theverge.com", "verge.com",
    "wired.com", "arstechnica.com", "infoworld.com", "computerworld.com",
    "engadget.com", "gizmodo.com", "mashable.com", "thenextweb.com",
    "digitaltrends.com", "fastcompany.com", "techrepublic.com",
    "theregister.com", "siliconangle.com", "readwrite.com", "geekwire.com",
    "cmswire.com", "slashdot.org", "technologyreview.com", "theinformation.com",
    "gigaom.com", "makeuseof.com", "techspot.com", "tomshardware.com",
    "9to5mac.com", "bgr.com", "techdirt.com", "hackaday.com",
    "informationweek.com", "pcworld.com", "extremetech.com",
    "siliconrepublic.com", "geekflare.com", "rtings.com",
    "forbes.com", "businessinsider.com", "entrepreneur.com", "inc.com",
    "nytimes.com", "wsj.com", "bloomberg.com", "reuters.com",
    "fortune.com", "economist.com", "adweek.com", "marketwatch.com",
    "cnbc.com", "ft.com", "axios.com", "businesswire.com", "prnewswire.com",
    "globenewswire.com",
    "adage.com", "digiday.com", "marketingdive.com", "adexchanger.com",
    "martech.org", "thedrum.com", "mediapost.com", "chiefmarketer.com",
    "marketingweek.com", "searchengineland.com", "searchenginejournal.com",
    "marketingbrew.com", "brandingmag.com", "martechseries.com",
    "socialmediatoday.com", "contentmarketinginstitute.com",
    "seroundtable.com", "moz.com",
    "darkreading.com", "securityweek.com", "thehackernews.com",
    "krebsonsecurity.com", "csoonline.com", "threatpost.com",
    "hackread.com", "infosecurity-magazine.com", "bleepingcomputer.com",
    "cybersecuritydive.com",
    "thenewstack.io", "devops.com", "cloudnativenow.com",
    "containerjournal.com", "cloudwards.net",
    "cio.com", "ciodive.com", "techtarget.com",
    "hbr.org", "mckinsey.com", "bain.com", "bcg.com", "deloitte.com",
    "accenture.com", "pwc.com", "kpmg.com", "ey.com",
    "oliverwyman.com", "rolandberger.com", "451research.com", "omdia.com",
    "constellationr.com", "everestgrp.com", "frost.com", "nucleusresearch.com",
    "redmonk.com", "hfsresearch.com", "canalys.com", "verdantix.com",
    "abiresearch.com", "globaldata.com",
    "retaildive.com", "supplychaindive.com", "hrdive.com",
    "constructiondive.com", "fooddive.com", "grocerydive.com",
    "healthcaredive.com", "manufacturingdive.com", "utilitydive.com",
    "restaurantdive.com", "bankingdive.com", "biopharmadive.com",
    "paymentsdive.com", "hoteldive.com", "wastedive.com",
    "ecommercetimes.com", "digitalcommerce360.com", "healthcareitnews.com",
    "fintechfutures.com", "pymnts.com", "businessofapps.com",
    "supplychaindigital.com", "fintechmagazine.com",
    "fintechweekly.com", "thefintechtimes.com", "fintech.global",
    "shrm.org",
    "kdnuggets.com", "aimagazine.com",
    "wikipedia.org", "reddit.com", "quora.com", "stackexchange.com",
    "stackoverflow.com", "medium.com", "substack.com",
    "youtube.com", "producthunt.com", "crunchbase.com",
    "dev.to", "github.com", "hashnode.com", "codeproject.com",
    "hackernoon.com", "dzone.com", "sitepoint.com", "smashingmagazine.com",
    "freecodecamp.org",
    "news.google.com", "news.yahoo.com", "msn.com", "flipboard.com",
    "smartnews.com", "newsbreak.com", "feedly.com", "allsides.com",
    "techmeme.com", "hacker-news.firebaseio.com",
    "accesswire.com", "prweb.com",
    "angel.co", "wellfound.com", "startupgrind.com",
    "glassdoor.com", "builtin.com",
    "dribbble.com", "behance.net", "awwwards.com", "designrush.com",
    "webdesignerdepot.com",
    "hostingadvice.com", "wpbeginner.com", "top10.com",
    "britannica.com", "investopedia.com", "coursereport.com",
    "zapier.com", "omr.com", "thedigitalprojectmanager.com", "toptal.com",
    "european-alternatives.eu", "softwaretestingmaterial.com",
    "thectoclub.com", "proposal.biz", "easyreplenish.com",
    "softwareworld.co", "appsumo.com", "killerstartups.com",
    "wirecutter.com", "consumerreports.org",
}

STRUCTURED_DATA_TYPES: set[str] = {
    "faqpage", "faq", "product", "howto", "softwareapplication",
    "article", "blogposting", "review", "aggregaterating",
    "offer", "itemlist", "breadcrumblist", "videoobject",
    "dataset", "course", "event", "recipe", "qapage",
}

AUTHORITY_SUFFIXES: set[str] = {"edu", "gov", "gov.uk", "ac.uk", "mil"}
AUTHORITY_DOMAINS: set[str] = {
    "wikipedia.org", "scholar.google.com", "ncbi.nlm.nih.gov",
    "arxiv.org", "nature.com", "sciencedirect.com", "ieee.org",
    "acm.org", "researchgate.net", "pubmed.ncbi.nlm.nih.gov",
    "springer.com", "wiley.com", "jstor.org", "ssrn.com",
    "nber.org", "who.int", "un.org", "worldbank.org",
    "statista.com", "pewresearch.org", "gallup.com",
}

LINK_FILTER_DOMAINS: set[str] = {
    "facebook.com", "twitter.com", "x.com", "linkedin.com", "instagram.com",
    "pinterest.com", "tiktok.com", "youtube.com", "apple.com",
    "play.google.com", "apps.apple.com",
    "cdn.jsdelivr.net", "cdnjs.cloudflare.com", "fonts.googleapis.com",
    "ajax.googleapis.com", "maxcdn.bootstrapcdn.com",
}


# ── Patterns (verbatim from extract_features.py) ─────────────────────────────

_STAT_PATTERNS = [
    re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"),
    re.compile(r"\b\d+\.?\d*%"),
    re.compile(r"\b(?:19|20)\d{2}\b"),
    re.compile(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"),
    re.compile(r"\$\d+(?:,\d{3})*(?:\.\d{2})?"),
    re.compile(r"\b\d+(?:\.\d+)?[BMKbmk]\b"),
]

_QUESTION_RE = re.compile(
    r"^\s*(?:what|how|why|when|where|which|who|can|does|is|are|should|will|do)\b",
    re.IGNORECASE,
)

_DATE_PATTERNS = [
    re.compile(r"(\d{4}-\d{2}-\d{2})"),
    re.compile(r"(\d{4}/\d{2}/\d{2})"),
    re.compile(r"(\w+ \d{1,2},?\s*\d{4})", re.IGNORECASE),
    re.compile(r"(\d{1,2} \w+ \d{4})", re.IGNORECASE),
]

_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%B %d %Y",
    "%d %B %Y", "%b %d, %Y", "%b %d %Y", "%d %b %Y",
]


# ── HTML helpers (slimmed; HTMLLoader handles cache IO) ──────────────────────

def _get_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _extract_body_text(soup: BeautifulSoup) -> str:
    soup = copy(soup)
    for tag in soup.find_all(["script", "style", "nav", "footer", "header",
                              "noscript", "iframe", "svg"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()
    body = soup.find("body")
    if body is None:
        return ""
    return body.get_text(separator=" ", strip=True)


# ── Treatment extractors (verbatim ports) ────────────────────────────────────

def extract_t1a_stats_present(body_text: str) -> int:
    if not body_text.strip():
        return 0
    return int(any(p.search(body_text) for p in _STAT_PATTERNS))


def extract_t1b_stats_density(body_text: str) -> float | None:
    if not body_text.strip():
        return None
    words = body_text.split()
    word_count = len(words)
    if word_count == 0:
        return None
    found: set[str] = set()
    for pat in _STAT_PATTERNS:
        for m in pat.finditer(body_text):
            found.add(m.group())
    return round(len(found) / (word_count / 500), 2)


def extract_t2a_question_headings(soup: BeautifulSoup) -> int:
    for heading in soup.find_all(["h2", "h3"]):
        text = heading.get_text(strip=True)
        if _QUESTION_RE.match(text) or text.endswith("?"):
            return 1
    return 0


def extract_t2b_structural_modularity(soup: BeautifulSoup) -> int:
    return len(soup.find_all(["h2", "h3"]))


def _check_ld_type(data: Any, target_types: set[str]) -> bool:
    if isinstance(data, dict):
        type_val = data.get("@type", "")
        if isinstance(type_val, str) and type_val.lower() in target_types:
            return True
        if isinstance(type_val, list):
            if any(t.lower() in target_types for t in type_val if isinstance(t, str)):
                return True
        if "@graph" in data:
            return _check_ld_type(data["@graph"], target_types)
    elif isinstance(data, list):
        return any(_check_ld_type(item, target_types) for item in data)
    return False


def extract_t3_structured_data(soup: BeautifulSoup) -> int:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        if _check_ld_type(data, STRUCTURED_DATA_TYPES):
            return 1
    return 0


def _link_domain(href: str) -> str:
    """Extract the registrable domain from an href; '' if non-http(s)."""
    try:
        parsed = urlparse(href)
        if parsed.scheme not in ("http", "https"):
            return ""
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        # Crude registrable-domain heuristic. Upstream uses tldextract; we
        # avoid the optional dep. For known multi-part suffixes (gov.uk,
        # ac.uk) the AUTHORITY_SUFFIXES check handles them via endswith.
        return host
    except Exception:
        return ""


def _link_suffix(host: str) -> str:
    """Return the TLD-side suffix (e.g. 'edu', 'gov', 'gov.uk')."""
    parts = host.rsplit(".", 2)
    if len(parts) >= 2:
        # Try two-part suffix first (gov.uk)
        if len(parts) >= 3:
            two = ".".join(parts[-2:])
            if two in AUTHORITY_SUFFIXES:
                return two
        return parts[-1]
    return ""


def extract_t4a_ext_citations_any(soup: BeautifulSoup, page_domain: str) -> int:
    page_domain = (page_domain or "").lower()
    for a in soup.find_all("a", href=True):
        host = _link_domain(a["href"])
        if not host or host == page_domain or host in LINK_FILTER_DOMAINS:
            continue
        # Strip subdomain for comparison (rough): take last 2 labels.
        parts = host.split(".")
        link_domain = ".".join(parts[-2:]) if len(parts) >= 2 else host
        if link_domain == page_domain:
            continue
        return 1
    return 0


def extract_t4b_auth_citations(soup: BeautifulSoup, page_domain: str) -> int:
    page_domain = (page_domain or "").lower()
    count = 0
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href in seen:
            continue
        seen.add(href)
        host = _link_domain(href)
        if not host or host == page_domain:
            continue
        suffix = _link_suffix(host)
        # crude registrable-domain
        parts = host.split(".")
        link_domain = ".".join(parts[-2:]) if len(parts) >= 2 else host
        if suffix in AUTHORITY_SUFFIXES or link_domain in AUTHORITY_DOMAINS:
            count += 1
    return count


def _parse_date_str(s: str) -> datetime | None:
    s = s.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def extract_t6_freshness(soup: BeautifulSoup, body_text: str) -> int:
    """Ordinal 0-4. See upstream extract_features.py for the bucket cutoffs."""
    now = datetime.now(timezone.utc)
    dates_found: list[datetime] = []

    for meta in soup.find_all("meta"):
        name = (meta.get("name", "") or meta.get("property", "") or "").lower()
        content = meta.get("content", "") or ""
        if any(dn in name for dn in ("date", "published", "modified", "time")):
            dt = _parse_date_str(content)
            if dt:
                dates_found.append(dt)

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            for key in ("datePublished", "dateModified", "dateCreated"):
                val = data.get(key, "")
                if val:
                    dt = _parse_date_str(str(val))
                    if dt:
                        dates_found.append(dt)

    for time_tag in soup.find_all("time"):
        dt_attr = time_tag.get("datetime", "")
        if dt_attr:
            dt = _parse_date_str(dt_attr)
            if dt:
                dates_found.append(dt)

    if not dates_found:
        for pat in _DATE_PATTERNS:
            for m in pat.finditer(body_text[:5000]):
                dt = _parse_date_str(m.group(1))
                if dt and dt.year >= 2015 and dt <= now:
                    dates_found.append(dt)
                    break
            if dates_found:
                break

    if not dates_found:
        return 0

    most_recent = max(dates_found)
    age_days = (now - most_recent).days
    if age_days < 0:
        return 4
    if age_days <= 180:
        return 4
    if age_days <= 365:
        return 3
    if age_days <= 730:
        return 2
    if age_days <= 1825:
        return 1
    return 0


def classify_source_type(domain: str) -> tuple[int, int, str]:
    d = (domain or "").lower().strip()
    if d in BRAND_DOMAINS:
        return 1, 0, "brand"
    if d in EARNED_DOMAINS:
        return 0, 1, "earned"
    return 0, 0, "other"


# ── Confounders ──────────────────────────────────────────────────────────────

def conf_title_has_kw(title: str, keyword: str) -> int:
    if not title or not keyword:
        return 0
    kw_words = [w.lower() for w in keyword.split() if len(w) >= 3]
    title_lower = title.lower()
    return int(any(w in title_lower for w in kw_words))


def conf_brand_recog(domain: str) -> int:
    return 1 if (domain or "").lower().strip() in BRAND_DOMAINS else 0


def extract_word_count(body_text: str) -> int | None:
    if not body_text.strip():
        return None
    return len(body_text.split())


def extract_readability(body_text: str) -> float | None:
    if not body_text.strip() or len(body_text.split()) < 100:
        return None
    try:
        import textstat
        return round(textstat.flesch_kincaid_grade(body_text), 2)
    except Exception:
        return None


def extract_internal_links(soup: BeautifulSoup, page_domain: str) -> int:
    page_domain = (page_domain or "").lower()
    count = 0
    for a in soup.find_all("a", href=True):
        href = a["href"]
        parsed = urlparse(href)
        if not parsed.scheme and not parsed.netloc:
            if href.startswith(("/", "#", "?")):
                count += 1
            continue
        if parsed.scheme not in ("http", "https"):
            continue
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        parts = host.split(".")
        link_domain = ".".join(parts[-2:]) if len(parts) >= 2 else host
        if link_domain == page_domain:
            count += 1
    return count


def extract_outbound_links(soup: BeautifulSoup, page_domain: str) -> int:
    page_domain = (page_domain or "").lower()
    count = 0
    for a in soup.find_all("a", href=True):
        host = _link_domain(a["href"])
        if not host:
            continue
        parts = host.split(".")
        link_domain = ".".join(parts[-2:]) if len(parts) >= 2 else host
        if link_domain and link_domain != page_domain:
            count += 1
    return count


def extract_images_alt(soup: BeautifulSoup) -> int:
    return sum(1 for img in soup.find_all("img") if (img.get("alt", "") or "").strip())


# ── Embeddings + BM25 (semantic confounders, T5 topical_comp) ────────────────

def compute_embeddings(texts: list[str], model, batch_size: int = 64) -> np.ndarray:
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        out.append(embs)
    return np.vstack(out) if out else np.empty((0, 0))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_bm25_scores(keyword: str, page_texts: list[str]) -> list[float]:
    from rank_bm25 import BM25Okapi
    tokenized: list[list[str]] = []
    for text in page_texts:
        if text and text.strip():
            tokenized.append(text.lower().split()[:5000])
        else:
            tokenized.append([""])
    if not tokenized:
        return []
    bm25 = BM25Okapi(tokenized)
    return [float(s) for s in bm25.get_scores(keyword.lower().split())]


# ── Per-URL feature row ──────────────────────────────────────────────────────

_BASE_FEATURE_COLS = (
    "treat_stats_present", "treat_stats_density",
    "treat_question_headings", "treat_structural_modularity",
    "treat_structured_data", "treat_ext_citations_any", "treat_auth_citations",
    "treat_freshness",
    "treat_brand", "treat_earned", "treat_source_earned",
    "source_type",
    "conf_title_has_kw", "conf_brand_recog",
    "conf_title_len", "conf_snippet_len",
    "conf_word_count", "conf_readability",
    "conf_internal_links", "conf_outbound_links", "conf_images_alt",
    "conf_https",
    "html_present", "extract_error",
)


def _empty_row(keyword: str, url: str, position: int, title: str, snippet: str) -> dict:
    domain = _extract_domain(url)
    is_brand, is_earned, src = classify_source_type(domain)
    return {
        "keyword": keyword,
        "url": url,
        "domain": domain,
        "position": int(position) if position is not None else None,
        "title": title or "",
        "snippet": snippet or "",
        "treat_stats_present":         None,
        "treat_stats_density":         None,
        "treat_question_headings":     None,
        "treat_structural_modularity": None,
        "treat_structured_data":       None,
        "treat_ext_citations_any":     None,
        "treat_auth_citations":        None,
        "treat_freshness":             None,
        "treat_brand":                 is_brand,
        "treat_earned":                is_earned,
        "treat_source_earned":         is_earned,  # alias used downstream
        "source_type":                 src,
        "conf_title_has_kw":           conf_title_has_kw(title, keyword),
        "conf_brand_recog":            conf_brand_recog(domain),
        "conf_title_len":              len(title or ""),
        "conf_snippet_len":            len(snippet or ""),
        "conf_word_count":             None,
        "conf_readability":            None,
        "conf_internal_links":         None,
        "conf_outbound_links":         None,
        "conf_images_alt":             None,
        "conf_https":                  int(url.lower().startswith("https://")) if url else 0,
        "html_present":                False,
        "extract_error":               None,
    }


def extract_one_page(
    keyword: str, position: int, url: str, title: str, snippet: str,
    html: str | None,
) -> dict:
    """Return a feature row for one (keyword, url) pair. ``html=None`` => sparse row."""
    row = _empty_row(keyword, url, position, title, snippet)
    if not html:
        return row
    try:
        soup = _get_soup(html)
        body = _extract_body_text(soup)
        domain = row["domain"]
        row.update({
            "treat_stats_present":         extract_t1a_stats_present(body),
            "treat_stats_density":         extract_t1b_stats_density(body),
            "treat_question_headings":     extract_t2a_question_headings(soup),
            "treat_structural_modularity": extract_t2b_structural_modularity(soup),
            "treat_structured_data":       extract_t3_structured_data(soup),
            "treat_ext_citations_any":     extract_t4a_ext_citations_any(soup, domain),
            "treat_auth_citations":        extract_t4b_auth_citations(soup, domain),
            "treat_freshness":             extract_t6_freshness(soup, body),
            "conf_word_count":             extract_word_count(body),
            "conf_readability":            extract_readability(body),
            "conf_internal_links":         extract_internal_links(soup, domain),
            "conf_outbound_links":         extract_outbound_links(soup, domain),
            "conf_images_alt":             extract_images_alt(soup),
            "html_present":                True,
        })
    except Exception as e:
        row["extract_error"] = f"{type(e).__name__}: {e}"
    return row


# ── HTML loader resolution ───────────────────────────────────────────────────

def _pick_html_run_id(root: Path, engine: str, pool: int) -> str | None:
    """Find any cached run_id for this (engine, pool) that has html_cache."""
    runs = root / "data" / "runs"
    if not runs.exists():
        return None
    candidates: list[str] = []
    for d in runs.iterdir():
        if not d.is_dir():
            continue
        # Only un-suffixed (original cached) runs have HTML.
        rid = d.name
        if rid.endswith((
            "_biased", "_neutral",
            "_biased_passage", "_neutral_passage",
            "_biased_rag", "_neutral_rag",
        )):
            continue
        if not rid.startswith(engine + "_"):
            continue
        if f"_serp{pool}_" not in rid:
            continue
        if (d / "phase2" / "html_cache.tar.gz").exists() or (d / "phase2" / "html_cache").is_dir():
            candidates.append(rid)
    return candidates[0] if candidates else None


# ── Topic / BM25 confounders + T5 ────────────────────────────────────────────

def _add_semantic_columns(rows: list[dict], keyword: str, body_texts: list[str],
                          embedder=None) -> None:
    """In-place: add conf_title_kw_sim, conf_snippet_kw_sim, conf_bm25, treat_topical_comp."""
    titles   = [r["title"]   for r in rows]
    snippets = [r["snippet"] for r in rows]

    if embedder is not None:
        try:
            kw_emb     = compute_embeddings([keyword], embedder)[0]
            t_emb      = compute_embeddings(titles,    embedder) if titles   else np.empty((0, 0))
            s_emb      = compute_embeddings(snippets,  embedder) if snippets else np.empty((0, 0))
            body_emb   = compute_embeddings(body_texts, embedder) if body_texts else np.empty((0, 0))
            for i, r in enumerate(rows):
                r["conf_title_kw_sim"]   = cosine_sim(kw_emb, t_emb[i])  if i < len(t_emb)   else 0.0
                r["conf_snippet_kw_sim"] = cosine_sim(kw_emb, s_emb[i])  if i < len(s_emb)   else 0.0
                r["treat_topical_comp"]  = cosine_sim(kw_emb, body_emb[i]) if i < len(body_emb) else 0.0
        except Exception as e:
            for r in rows:
                r.setdefault("conf_title_kw_sim", None)
                r.setdefault("conf_snippet_kw_sim", None)
                r.setdefault("treat_topical_comp", None)
            print(f"[features] embedding failed for kw={keyword!r}: {e}", flush=True)
    else:
        # No embedder available -> leave None; merge.py / DML will handle missingness.
        for r in rows:
            r.setdefault("conf_title_kw_sim", None)
            r.setdefault("conf_snippet_kw_sim", None)
            r.setdefault("treat_topical_comp", None)

    # BM25 always works (pure CPU, no model load).
    bm25 = compute_bm25_scores(keyword, body_texts) if body_texts else []
    for i, r in enumerate(rows):
        r["conf_bm25"] = bm25[i] if i < len(bm25) else 0.0


def _maybe_load_embedder(device: str | None) -> Any | None:
    """Try to load all-MiniLM-L6-v2 (offline-capable). Return None on failure."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[features] sentence-transformers not installed; cosine columns disabled",
              flush=True)
        return None
    try:
        kw = {}
        if device:
            kw["device"] = device
        model = SentenceTransformer("all-MiniLM-L6-v2", **kw)
        print(f"[features] embedder loaded (device={getattr(model, 'device', 'cpu')})",
              flush=True)
        return model
    except Exception as e:
        print(f"[features] could not load embedder: {e}; cosine columns disabled",
              flush=True)
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage B: deterministic features from cached HTML.",
    )
    ap.add_argument("--engine", required=True, choices=C.ENGINES)
    ap.add_argument("--pool",   required=True, type=int, choices=(20, 50))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default=os.getenv("FEATURES_DEVICE"),
                    help="Embedder device: cpu, cuda, mps. Default: auto.")
    ap.add_argument("--no-embed", action="store_true",
                    help="Skip semantic-similarity columns (T5, conf_*_kw_sim).")
    ap.add_argument("--max-keywords", type=int, default=None,
                    help="Cap (smoke testing).")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--html-run-id", default=None,
                    help="Override the run_id whose html_cache.tar.gz to read from. "
                         "If unset, picks any matching (engine, pool) cached run.")
    args = ap.parse_args()

    root = data_root(args.data_root)

    feat_dir = root / "data" / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    out_path  = feat_dir / f"features_{args.engine}_top{args.pool}.parquet"
    ckpt_path = feat_dir / f".features_{args.engine}_top{args.pool}_ckpt.json"

    html_run = args.html_run_id or _pick_html_run_id(root, args.engine, args.pool)
    if html_run is None:
        print(f"[features] FATAL: no cached html_cache for engine={args.engine} pool={args.pool}",
              file=sys.stderr)
        return 2
    print(f"[features] html source run: {html_run}", flush=True)

    serp = load_serp(backend=args.engine, pool=args.pool, root=root)

    ckpt = Checkpoint.load(ckpt_path)
    done_keys: set[tuple[str, str]] = set()
    for k in ckpt.data.get("seen", []):
        kw, url = k.split("|||", 1)
        done_keys.add((kw, url))

    existing_rows: list[dict] = []
    if args.resume and out_path.exists():
        existing_rows = pd.read_parquet(out_path).to_dict(orient="records")
        for r in existing_rows:
            done_keys.add((r["keyword"], r["url"]))
        print(f"[features] resuming: {len(existing_rows):,} rows already present",
              flush=True)
    else:
        out_path.unlink(missing_ok=True)

    embedder = None
    if not args.no_embed:
        embedder = _maybe_load_embedder(args.device)

    new_rows: list[dict] = []
    n_seen_kws = 0

    loader = HTMLLoader(html_run, root=root)
    try:
        for kw, g in tqdm(serp.groupby("keyword", sort=False),
                          total=serp["keyword"].nunique(),
                          desc=f"features {args.engine}_top{args.pool}"):
            if args.max_keywords is not None and n_seen_kws >= args.max_keywords:
                break
            n_seen_kws += 1

            g = g.sort_values("position").head(args.pool)
            kw_rows: list[dict] = []
            kw_bodies: list[str] = []

            for _, r in g.iterrows():
                url = str(r.get("url", "") or "")
                if (kw, url) in done_keys:
                    continue
                title   = str(r.get("title", "") or "")
                snippet = str(r.get("snippet", "") or "")
                html = loader.get_html(url)
                row = extract_one_page(kw, int(r["position"]), url, title, snippet, html)
                # body text is needed for BM25/embedding; recompute cheaply rather
                # than carrying it through extract_one_page.
                body = ""
                if html and row["html_present"] and not row["extract_error"]:
                    try:
                        body = _extract_body_text(_get_soup(html))
                    except Exception:
                        body = ""
                kw_rows.append(row)
                kw_bodies.append(body)

            if not kw_rows:
                continue

            _add_semantic_columns(kw_rows, kw, kw_bodies, embedder=embedder)

            new_rows.extend(kw_rows)
            for r in kw_rows:
                ckpt.mark(f"{r['keyword']}|||{r['url']}")
            if len(new_rows) >= 200:
                # periodic flush
                df = pd.DataFrame(existing_rows + new_rows)
                df.to_parquet(out_path, index=False)
                ckpt.save()
    finally:
        loader.close()

    # Final flush
    df = pd.DataFrame(existing_rows + new_rows)
    df.to_parquet(out_path, index=False)
    ckpt.save()
    print(f"[features] wrote {len(df):,} rows -> {out_path}", flush=True)
    print(f"[features] new this run: {len(new_rows):,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
