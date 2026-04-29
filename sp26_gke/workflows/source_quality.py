"""
Domain-based source-quality scoring for research evidence.

Soft-scores every evidence URL into a tier (reputable/neutral/low) and hard-blocks
invalid URLs and domains on an explicit blocklist. Scoring is deterministic and depends
only on the registered domain and TLD.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

# Accept only the characters legally allowed in DNS hostnames or IP literals.
_VALID_HOST_RE = re.compile(r"^[a-z0-9.\-:\[\]]+$")

# Trusted top-level domains treated as reputable
AUTHORITATIVE_TLDS: frozenset[str] = frozenset({"gov", "edu", "mil", "int"})

# academic/government suffixes (e.g. cam.ac.uk, nih.go.jp).
AUTHORITATIVE_SECOND_LEVELS: frozenset[str] = frozenset(
    {
        "ac.uk",
        "gov.uk",
        "nhs.uk",
        "ac.jp",
        "go.jp",
        "edu.au",
        "gov.au",
        "edu.cn",
        "gov.cn",
        "ac.in",
        "gov.in",
        "edu.sg",
        "gov.sg",
    }
)

# allowlist of reputable organizations
ALLOWLIST_DOMAINS: frozenset[str] = frozenset(
    {
        "nature.com",
        "science.org",
        "sciencemag.org",
        "nejm.org",
        "thelancet.com",
        "cell.com",
        "ieee.org",
        "acm.org",
        "arxiv.org",
        "who.int",
        "europa.eu",
        "oecd.org",
        "worldbank.org",
        "imf.org",
        "un.org",
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "bbc.co.uk",
        "ft.com",
        "economist.com",
        "wsj.com",
        "nytimes.com",
        "bloomberg.com",
    }
)

# Domains hard-dropped before evidence is persisted.
# Kept empty for the current POC - can add to later
BLOCKLIST_DOMAINS: frozenset[str] = frozenset()

# user-generated / aggregator domains kept but tagged as low quality.
LOW_QUALITY_DOMAINS: frozenset[str] = frozenset(
    {
        "youtube.com",
        "youtu.be",
        "reddit.com",
        "quora.com",
        "pinterest.com",
        "answers.com",
        "wikihow.com",
        "ehow.com",
        "yahoo.com",
        "tumblr.com",
        "medium.com",
        "substack.com",
    }
)

# Multi-label public suffixes recognized when extracting registered domains.
_PUBLIC_SUFFIXES_2: frozenset[str] = frozenset(
    {
        "co.uk",
        "ac.uk",
        "gov.uk",
        "org.uk",
        "nhs.uk",
        "co.jp",
        "ac.jp",
        "go.jp",
        "or.jp",
        "com.au",
        "edu.au",
        "gov.au",
        "org.au",
        "net.au",
        "com.cn",
        "edu.cn",
        "gov.cn",
        "org.cn",
        "co.in",
        "ac.in",
        "gov.in",
        "co.nz",
        "ac.nz",
        "govt.nz",
        "com.br",
        "com.sg",
        "edu.sg",
        "gov.sg",
        "co.za",
        "ac.za",
        "co.kr",
    }
)


@dataclass(frozen=True)
class SourceQuality:
    """Per-URL quality signal attached to an Evidence object."""

    score: float
    tier: str
    reason: str
    blocked: bool


def extract_registered_domain(url: str) -> str:
    """
    Return the lowercase registered domain (e.g. ``cdc.gov``, ``bar.co.uk``).

    Returns an empty string for URLs without a parseable host. Strips port
    numbers and ``www.`` prefixes, and respects the small set of multi-label
    public suffixes in ``_PUBLIC_SUFFIXES_2``.
    """
    if not url:
        return ""

    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = (parsed.hostname or "").lower().strip()
    if not host or not _VALID_HOST_RE.match(host):
        return ""

    # Bare IP literals (v4 or bracketed v6) cannot be reduced further.
    if host.replace(".", "").isdigit() or host.startswith("["):
        return host

    labels = host.split(".")
    if len(labels) <= 2:
        return host

    last_two = ".".join(labels[-2:])
    if last_two in _PUBLIC_SUFFIXES_2 and len(labels) >= 3:
        return ".".join(labels[-3:])
    return last_two


def _domain_has_authoritative_suffix(domain: str) -> bool:
    if not domain:
        return False
    tld = domain.rsplit(".", 1)[-1]
    if tld in AUTHORITATIVE_TLDS:
        return True
    parts = domain.split(".")
    if len(parts) >= 2 and ".".join(parts[-2:]) in AUTHORITATIVE_SECOND_LEVELS:
        return True
    return False


def score_url(url: str) -> SourceQuality:
    """
    Score a URL into a quality tier using domain-policy precedence.

    Precedence: invalid -> blocklist -> allowlist -> trusted TLD ->
    low-quality domain -> neutral default. Output is deterministic for a given
    URL and policy.
    """
    domain = extract_registered_domain(url)
    if not domain:
        return SourceQuality(
            score=0.0, tier="blocked", reason="invalid_url", blocked=True
        )

    if domain in BLOCKLIST_DOMAINS:
        return SourceQuality(
            score=0.0, tier="blocked", reason="blocklist", blocked=True
        )

    if domain in ALLOWLIST_DOMAINS:
        return SourceQuality(
            score=1.0, tier="reputable", reason="allowlist", blocked=False
        )

    if _domain_has_authoritative_suffix(domain):
        return SourceQuality(
            score=0.9,
            tier="reputable",
            reason="authoritative_tld",
            blocked=False,
        )

    if domain in LOW_QUALITY_DOMAINS:
        return SourceQuality(
            score=0.25, tier="low", reason="low_quality_domain", blocked=False
        )

    return SourceQuality(score=0.5, tier="neutral", reason="default", blocked=False)
