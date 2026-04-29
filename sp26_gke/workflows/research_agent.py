"""
Deep Research Agent — LangGraph + Gemini with grounded web search.

Based on: https://towardsdatascience.com/langgraph-101-lets-build-a-deep-research-agent/
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import operator
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime
from typing import Annotated, Any, TypedDict

import google.genai as genai
import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field, field_validator

from sp26_gke.workflows.source_quality import score_url

try:
    from langgraph.checkpoint.postgres import (  # type: ignore[import-not-found]
        PostgresSaver as _PostgresSaver,
    )
except ImportError:
    _PostgresSaver = None

PostgresSaver: Any | None = _PostgresSaver

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────


class Configuration(BaseModel):
    query_generator_model: str = Field(default="gemini-2.5-flash")
    reflection_model: str = Field(default="gemini-2.5-flash")
    answer_model: str = Field(default="gemini-2.5-flash")
    number_of_initial_queries: int = Field(default=1)
    max_research_loops: int = Field(default=2)
    max_citations_per_search: int = Field(default=5)
    max_queries_per_run: int = Field(default=7)
    allowed_evidence_tiers: str = Field(default="reputable,neutral")
    min_allowed_evidence_total: int = Field(default=5)
    min_allowed_evidence_per_section: int = Field(default=1)
    min_reputable_evidence_total: int = Field(default=1)
    provider_retry_attempts: int = Field(default=2)
    provider_retry_backoff_seconds: float = Field(default=3.0)

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> Configuration:
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        field_names = list(Configuration.model_fields)
        values: dict[str, Any] = {
            k: os.environ.get(k.upper(), configurable.get(k)) for k in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})


# ── State ────────────────────────────────────────────────────────────────────


def merge_marker_sources(
    left: dict[int, list[dict[str, str]]],
    right: dict[int, list[dict[str, str]]],
) -> dict[int, list[dict[str, str]]]:
    """Reducer for merging marker_id -> sources maps across parallel branches."""
    out: dict[int, list[dict[str, str]]] = {k: list(v) for k, v in left.items()}
    for marker_id, sources in right.items():
        if marker_id not in out:
            out[marker_id] = list(sources)
            continue

        combined = out[marker_id] + list(sources)
        deduped_by_url: dict[str, dict[str, str]] = {}
        for src in combined:
            url = src.get("url", "")
            if url:
                # Keep the first occurrence of each URL for stability.
                deduped_by_url.setdefault(url, src)
            else:
                # If the source has no URL, keep it but don't attempt deduplication.
                deduped_by_url[str(len(deduped_by_url))] = src

        out[marker_id] = list(deduped_by_url.values())
    return out


def merge_section_lists(
    left: dict[str, list[str]], right: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Reducer for merging section_id -> string-list maps across branches."""
    out: dict[str, list[str]] = {k: list(v) for k, v in left.items()}
    for section_id, items in right.items():
        out.setdefault(section_id, [])
        out[section_id].extend(items)
    return out


def merge_section_marker_sources(
    left: dict[str, dict[int, list[dict[str, str]]]],
    right: dict[str, dict[int, list[dict[str, str]]]],
) -> dict[str, dict[int, list[dict[str, str]]]]:
    """Reducer for section-scoped marker sources."""
    out: dict[str, dict[int, list[dict[str, str]]]] = {
        section_id: {marker: list(srcs) for marker, srcs in marker_map.items()}
        for section_id, marker_map in left.items()
    }
    for section_id, marker_map in right.items():
        section_out = out.setdefault(section_id, {})
        for marker_id, sources in marker_map.items():
            section_out.setdefault(marker_id, [])
            section_out[marker_id].extend(sources)
    return out


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: ResearchPlan | None
    section_order: list[str]
    section_results: Annotated[dict[str, list[str]], merge_section_lists]
    section_evidence: Annotated[dict[str, list[dict]], merge_section_lists]
    section_allowed_results: Annotated[dict[str, list[str]], merge_section_lists]
    section_allowed_evidence: Annotated[dict[str, list[dict]], merge_section_lists]
    section_queries: Annotated[dict[str, list[str]], merge_section_lists]
    section_marker_sources: Annotated[
        dict[str, dict[int, list[dict[str, str]]]],
        merge_section_marker_sources,
    ]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    marker_sources: Annotated[dict[int, list[dict[str, str]]], merge_marker_sources]
    evidence_extraction_events: Annotated[list[str], operator.add]
    quality_gate_events: Annotated[list[str], operator.add]
    paragraph_evidence_mappings: Annotated[list[dict[str, Any]], operator.add]
    unsupported_paragraphs: Annotated[list[dict[str, str]], operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str


class QueryGenerationState(TypedDict):
    query_list: list[dict[str, str]]


class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    quality_gate_met: bool
    quality_gap_sections: list[str]
    quality_gap: str
    reputable_count: int
    reputable_shortfall: int
    research_loop_count: int
    number_of_ran_queries: int


class WebSearchState(TypedDict):
    search_query: str
    id: str
    section_id: str


# ── Schemas ──────────────────────────────────────────────────────────────────


class SearchQueryList(BaseModel):
    query: list[str] = Field(description="A list of search queries for web research.")
    rationale: str = Field(description="Why these queries are relevant.")


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether gathered info is sufficient to answer the topic."
    )
    knowledge_gap: str = Field(description="What information is missing.")
    follow_up_queries: list[str] = Field(
        default_factory=list, description="Follow-up search queries."
    )


class Evidence(BaseModel):
    query_plan_id: str = Field(
        description="The section id from the research plan this evidence belongs to"
    )
    claim: str = Field(
        description="A single complete-sentence claim extracted from the source. "
        "Do not merge multiple claims into one."
    )
    source_url: str = Field(description="The URL this claim came from")
    retrieval_query: str = Field(
        description="The search query that surfaced this source"
    )
    source_quality_score: float = Field(
        default=0.5,
        description="Domain-policy quality score in [0, 1]; populated post-extraction.",
    )
    source_quality_tier: str = Field(
        default="neutral",
        description="Quality tier label (reputable/neutral/low/blocked).",
    )
    source_quality_reason: str = Field(
        default="default",
        description="Reason the tier was assigned (allowlist/authoritative_tld/etc).",
    )

    @field_validator("claim")
    @classmethod
    def claim_not_empty(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("Claim too short to be a meaningful evidence object")
        return v.strip()

    @field_validator("source_url")
    @classmethod
    def valid_url(cls, v: str) -> str:
        if not v.startswith("http"):
            raise ValueError("source_url must be a valid URL")
        return v


class EvidenceList(BaseModel):
    items: list[Evidence] = Field(
        description="All atomic evidence objects extracted from this web research step. "
        "One object per claim per source."
    )


class MappedParagraph(BaseModel):
    text: str = Field(description="Paragraph text for this section.")
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence IDs from the section payload that support this paragraph.",
    )


class MappedSection(BaseModel):
    section_id: str = Field(description="Section id this content belongs to.")
    paragraphs: list[MappedParagraph] = Field(default_factory=list)


class MappedReport(BaseModel):
    sections: list[MappedSection] = Field(default_factory=list)


class PlanSection(BaseModel):
    id: str = Field(description="Stable section identifier in snake_case.")
    title: str = Field(description="Human-readable section heading.")
    goal: str = Field(description="What this section should establish.")
    key_questions: list[str] = Field(
        default_factory=list, description="Key research questions for this section."
    )
    query_hints: list[str] = Field(
        default_factory=list, description="Helpful seed terms for search query writing."
    )
    required: bool = Field(
        default=False, description="Whether this section is required in the report."
    )


class ResearchPlan(BaseModel):
    topic_rewrite: str = Field(
        description="Optional clarified rewrite of the original research topic."
    )
    overall_success_criteria: str = Field(
        description="How to judge whether this research run is complete."
    )
    sections: list[PlanSection] = Field(
        default_factory=list, description="Ordered plan sections for this topic."
    )


# ── Prompts ──────────────────────────────────────────────────────────────────


def _current_date() -> str:
    return datetime.now().strftime("%B %d, %Y")


def _get_research_topic(messages: list) -> str:
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg["content"]
    return str(messages[0]) if messages else ""


PLANNER_PROMPT = """Create a topic-adaptive research plan for this technical research task.
Current date: {current_date}
Topic: {research_topic}

Build an ordered plan with 4-8 sections and these rules:
- Preserve intent for required rigor sections:
  - Executive_summary
  - Scope_and_definitions
  - Findings (with topic-adaptive sub-areas)
  - Evidence_and_credibility_notes
  - Open_questions_and_gaps
- Add domain-specific sections only when they are warranted by the topic.
- For each section include: id (snake_case), title, goal, 3-6 key_questions, 2-4 query_hints, and required.

Respond as JSON matching the provided schema exactly."""


QUERY_WRITER_PROMPT = """Generate {number_queries} diverse, targeted web search queries to research this topic.
Current date: {current_date}
Topic: {research_topic}

Rules:
- Prefer a single query unless the topic has multiple distinct aspects
- Queries must be specific and likely to return current, authoritative results
- Prefer institutional and primary sources by adding domain constraints where relevant
  (e.g. site:.gov, site:.edu, site:who.int, site:oecd.org, site:worldbank.org,
  site:un.org, site:europa.eu, or topic-specific standards bodies/journals)
- Avoid entertainment/video-first domains unless explicitly required by the topic
  (especially avoid youtube.com and youtu.be)
- No duplicate or near-duplicate queries

Respond as JSON with keys "rationale" (string) and "query" (list of strings)."""

SECTION_QUERY_WRITER_PROMPT = """Generate {number_queries} diverse, targeted web search queries for one section of a technical research report.
Current date: {current_date}
Topic: {research_topic}
Section id: {section_id}
Section title: {section_title}
Section goal: {section_goal}
Section key questions:
{section_key_questions}

Rules:
- Focus only on this section's scope.
- Queries must be specific and likely to return current, authoritative results.
- Prefer institutional and primary sources by adding domain constraints where relevant
  (e.g. site:.gov, site:.edu, site:who.int, site:oecd.org, site:worldbank.org,
  site:un.org, site:europa.eu, or topic-specific standards bodies/journals).
- Avoid entertainment/video-first domains unless explicitly required by the topic
  (especially avoid youtube.com and youtu.be).
- No duplicate or near-duplicate queries.

Respond as JSON with keys "rationale" (string) and "query" (list of strings)."""

WEB_SEARCHER_PROMPT = """You are a web research assistant. Search the web for accurate, up-to-date information on:
{research_topic}

Current date: {current_date}
Use the Google Search tool. Find factual, well-sourced information and cite your sources."""

REFLECTION_PROMPT = """Review this research and assess whether it fully answers the topic.
Current date: {current_date}
Topic: {research_topic}

Research gathered:
{summaries}

Determine: (1) is this sufficient? (2) what's missing? (3) what follow-up queries would help?"""

ANSWER_PROMPT = """Synthesize the following research into a comprehensive, well-structured answer.
Current date: {current_date}
Topic: {research_topic}

Planned section order:
{section_order}

Section-structured research payload:
{summaries}

Formatting contract:
- Use exactly these top-level headings in exactly this order: one heading per planned section.
- Heading format must be: ## [section_id] Section Title
- Do not add extra top-level headings.
- For each section, write only from that section's evidence snippets.
- If a section has weak or missing evidence, include a `Gaps:` subsection in that section.
- Use inline citations (e.g. [1], [2]) where relevant.

Before finalizing, perform an internal checklist:
1) All planned headings are present.
2) Headings are in the exact planned order.
3) No extra top-level headings were added.
Do not print the checklist."""

ANSWER_MAPPED_PROMPT = """Compose a sectioned research report from evidence cards.
Current date: {current_date}
Topic: {research_topic}

Planned section order:
{section_order}

Evidence payload:
{summaries}

Hard rules:
- Output must match the provided structured schema exactly.
- Use only evidence IDs provided for that section.
- Do not invent evidence IDs.
- Each paragraph should be grounded to one or more evidence IDs when possible.
- Keep paragraphs concise and factual.
- Do not include citation markers like [1] in paragraph text.
"""


# ── Citation helpers ─────────────────────────────────────────────────────────

SourceRef = dict[str, str]


class _CitationEntry(TypedDict):
    end: int
    sources: list[SourceRef]


def _content_to_text(content: Any) -> str:
    """Best-effort conversion of LangChain/Gemini message content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                # Common multimodal message formats store text under keys like "text" or "content".
                text_val = part.get("text") or part.get("content") or ""
                parts.append(str(text_val) if text_val else str(part))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def _extract_sources(
    response: Any,
    *,
    citation_base: int = 0,
    max_citations_per_search: int = 5,
) -> tuple[dict[int, list[SourceRef]], str]:
    """
    Return (marker_sources_map, text_with_citation_markers) from Gemini grounded output.

    The returned `marker_sources_map` is keyed by stable citation marker id,
    ensuring marker ids can be mapped back to URL(s) later.
    """
    text = response.text or ""
    marker_sources: dict[int, list[SourceRef]] = {}
    try:
        meta = response.candidates[0].grounding_metadata
        chunks = getattr(meta, "grounding_chunks", []) or []
        supports = getattr(meta, "grounding_supports", []) or []

        citations: list[_CitationEntry] = []
        for support in supports:
            seg = getattr(support, "segment", None)
            if not seg:
                continue
            indices = getattr(support, "grounding_chunk_indices", [])
            segs: list[SourceRef] = []
            for idx in indices:
                if idx < len(chunks):
                    web = getattr(chunks[idx], "web", None)
                    if web:
                        segs.append(
                            {
                                "url": str(getattr(web, "uri", "") or ""),
                                "title": str(getattr(web, "title", "") or ""),
                            }
                        )
            if segs:
                citations.append(
                    {
                        "end": int(getattr(seg, "end_index", 0) or 0),
                        "sources": segs,
                    }
                )

        sorted_citations = sorted(citations, key=lambda c: c["end"], reverse=True)[
            :max_citations_per_search
        ]
        for i, cite in enumerate(sorted_citations, 1):
            marker_id = citation_base + i
            end = cite["end"]
            text = text[:end] + f" [{marker_id}]" + text[end:]

            # De-dupe sources inside each marker by URL.
            deduped: dict[str, SourceRef] = {}
            for src in cite["sources"]:
                url = src.get("url", "")
                if url and url not in deduped:
                    deduped[url] = src
            marker_sources[marker_id] = list(deduped.values())

    except Exception:
        pass

    return marker_sources, text


def canonicalize_url(url: str) -> str:
    """Resolve Vertex redirect URLs to their real destination."""
    if "grounding-api-redirect" not in url and "vertexaisearch" not in url:
        return url
    try:
        with httpx.Client() as client:
            headers = {"User-Agent": "Mozilla/5.0"}
            try:
                r = client.head(
                    url, follow_redirects=True, timeout=5.0, headers=headers
                )
                resolved = str(r.url)
                if resolved and "grounding-api-redirect" not in resolved:
                    return resolved
            except Exception:
                pass
            r = client.get(url, follow_redirects=True, timeout=8.0, headers=headers)
            resolved = str(r.url)
            if resolved and "grounding-api-redirect" not in resolved:
                return resolved
            return resolved or url
    except Exception:
        return url


EVIDENCE_EXTRACTION_PROMPT = """Extract atomic evidence objects from the following web research text.

Section ID: {section_id}
Search query: {search_query}

Citation marker -> URL mapping:
{marker_url_mapping}

Web research text:
{text}

Rules:
- Extract one Evidence object per distinct factual claim per source.
- Do NOT summarize or rewrite claims — extract them as written in the text.
- Map each citation marker [N] back to its URL using the mapping above.
- If a claim has no citation marker, skip it.
- query_plan_id must be the section_id provided above.
- retrieval_query must be the search query provided above.

Respond as JSON matching the provided schema exactly."""


# ── Nodes ────────────────────────────────────────────────────────────────────

_genai_client: genai.Client | None = None
_genai_lock = threading.Lock()


def _get_genai_client() -> genai.Client:
    """
    Return a thread-safe singleton Gemini client.

    A single long-lived client avoids the 'client has been closed' error that occurs
    when short-lived clients are GC'd while parallel threads are mid-request through the
    shared httpx transport.
    """
    global _genai_client
    if _genai_client is not None:
        return _genai_client

    with _genai_lock:
        if _genai_client is not None:
            return _genai_client

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Set it in your environment (or .env) to run "
                "web research."
            )
        _genai_client = genai.Client(api_key=api_key)
        return _genai_client


def _make_llm(model: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )


def _invoke_with_provider_backoff[T](
    fn: Callable[[], T], *, attempts: int, backoff_seconds: float
) -> T:
    """Retry transient provider capacity failures with linear backoff."""
    bounded_attempts = max(1, attempts)
    bounded_backoff = max(0.1, backoff_seconds)
    for attempt in range(1, bounded_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            is_capacity_error = "503" in str(exc) or "UNAVAILABLE" in str(exc)
            if not is_capacity_error or attempt >= bounded_attempts:
                raise
            time.sleep(bounded_backoff * attempt)
    raise RuntimeError("Provider backoff failed unexpectedly")


def _plan_to_brief(plan: ResearchPlan) -> str:
    lines = [
        "Research plan:",
        f"- topic_rewrite: {plan.topic_rewrite}",
        f"- success_criteria: {plan.overall_success_criteria}",
    ]
    for idx, section in enumerate(plan.sections, 1):
        req = "required" if section.required else "optional"
        lines.append(f"{idx}. [{section.id}] {section.title} ({req})")
        lines.append(f"   goal: {section.goal}")
    return "\n".join(lines)


def _coerce_research_plan(plan_obj: Any) -> ResearchPlan | None:
    """Normalize stored plan payloads across reruns/reloads."""
    if plan_obj is None:
        return None
    if isinstance(plan_obj, ResearchPlan):
        return plan_obj
    if isinstance(plan_obj, dict):
        return ResearchPlan.model_validate(plan_obj)
    if isinstance(plan_obj, BaseModel):
        return ResearchPlan.model_validate(plan_obj.model_dump())
    raise TypeError(f"Unsupported plan payload type: {type(plan_obj)!r}")


def _evidence_id(section_id: str, evidence: dict[str, Any]) -> str:
    key = (
        f"{section_id}|{evidence.get('claim', '')}|{evidence.get('source_url', '')}|"
        f"{evidence.get('retrieval_query', '')}"
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _build_section_evidence_index(
    section_allowed_evidence: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for section_id, items in section_allowed_evidence.items():
        section_map: dict[str, dict[str, Any]] = {}
        for ev in items:
            if not isinstance(ev, dict):
                continue
            ev_id = _evidence_id(section_id, ev)
            section_map.setdefault(ev_id, ev)
        out[section_id] = section_map
    return out


def _build_source_title_lookup(
    marker_sources: dict[int, list[dict[str, str]]],
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for sources in marker_sources.values():
        for src in sources:
            raw_url = str(src.get("url", "")).strip()
            title = str(src.get("title", "")).strip()
            if not raw_url:
                continue
            canonical = canonicalize_url(raw_url)
            if canonical:
                lookup.setdefault(canonical, title or canonical)
            lookup.setdefault(raw_url, title or canonical or raw_url)
    return lookup


def _build_evidence_first_payload(
    *,
    plan: ResearchPlan | None,
    section_order: list[str],
    section_evidence_index: dict[str, dict[str, dict[str, Any]]],
    max_evidence_per_section: int = 8,
) -> str:
    section_map: dict[str, PlanSection] = {}
    if plan:
        section_map = {section.id: section for section in plan.sections}
    parts: list[str] = []
    for idx, section_id in enumerate(section_order, 1):
        section = section_map.get(section_id)
        title = section.title if section else section_id.replace("_", " ").title()
        goal = section.goal if section else "Summarize findings for this section."
        evidence_cards: list[str] = []
        for ev_id, ev in list(section_evidence_index.get(section_id, {}).items())[
            :max_evidence_per_section
        ]:
            claim = str(ev.get("claim", "")).replace("\n", " ").strip()
            source_url = str(ev.get("source_url", "")).strip()
            tier = str(ev.get("source_quality_tier", "neutral")).strip()
            score = float(ev.get("source_quality_score", 0.5))
            evidence_cards.append(
                f"- id={ev_id} | tier={tier} {score:.2f} | url={source_url} | claim={claim}"
            )
        evidence_text = "\n".join(evidence_cards) or "- (no evidence cards)"
        parts.append(
            "\n".join(
                [
                    f"SECTION {idx}",
                    f"id: {section_id}",
                    f"title: {title}",
                    f"goal: {goal}",
                    "evidence_cards:",
                    evidence_text,
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def _build_section_synthesis_payload(
    *,
    plan: ResearchPlan | None,
    section_order: list[str],
    section_results: dict[str, list[str]],
    fallback_summaries: list[str],
    max_snippets_per_section: int = 3,
) -> str:
    if not section_order:
        return "\n\n---\n\n".join(fallback_summaries)

    section_map: dict[str, PlanSection] = {}
    if plan:
        section_map = {section.id: section for section in plan.sections}

    parts: list[str] = []
    for idx, section_id in enumerate(section_order, 1):
        section = section_map.get(section_id)
        title = section.title if section else section_id.replace("_", " ").title()
        goal = section.goal if section else "Summarize findings for this section."
        snippets = section_results.get(section_id, [])[:max_snippets_per_section]
        snippets_text = (
            "\n".join(f"- {snippet}" for snippet in snippets) or "- (no evidence)"
        )
        parts.append(
            "\n".join(
                [
                    f"SECTION {idx}",
                    f"id: {section_id}",
                    f"title: {title}",
                    f"goal: {goal}",
                    "evidence_snippets:",
                    snippets_text,
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def _allowed_tier_set(cfg: Configuration) -> set[str]:
    raw = cfg.allowed_evidence_tiers.strip()
    if not raw:
        return {"reputable"}
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def _quality_gate_status(
    state: OverallState, cfg: Configuration
) -> tuple[bool, list[str], int, int]:
    allowed_by_section = state.get("section_allowed_evidence", {})
    section_evidence = state.get("section_evidence", {})
    section_order = state.get("section_order", [])
    total_allowed = sum(len(items) for items in allowed_by_section.values())
    total_reputable = 0
    for items in section_evidence.values():
        for ev in items:
            if str(ev.get("source_quality_tier", "")).lower() == "reputable":
                total_reputable += 1
    gap_sections: list[str] = []
    for section_id in section_order:
        count = len(allowed_by_section.get(section_id, []))
        if count < cfg.min_allowed_evidence_per_section:
            gap_sections.append(section_id)
    meets_total = total_allowed >= cfg.min_allowed_evidence_total
    meets_reputable = total_reputable >= cfg.min_reputable_evidence_total
    return (
        meets_total and not gap_sections and meets_reputable,
        gap_sections,
        total_allowed,
        total_reputable,
    )


def plan_research(state: OverallState, config: RunnableConfig) -> OverallState:
    cfg = Configuration.from_runnable_config(config)
    llm = _make_llm(cfg.query_generator_model)
    result = llm.with_structured_output(ResearchPlan).invoke(
        PLANNER_PROMPT.format(
            current_date=_current_date(),
            research_topic=_get_research_topic(state["messages"]),
        )
    )
    plan: ResearchPlan = ResearchPlan.model_validate(result)
    plan_brief = _plan_to_brief(plan)
    return {  # type: ignore[typeddict-item]
        "messages": [AIMessage(content=plan_brief)],
        "plan": plan,
        "section_order": [section.id for section in plan.sections],
    }


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    cfg = Configuration.from_runnable_config(config)
    count = state.get("initial_search_query_count") or cfg.number_of_initial_queries
    max_queries_per_run = max(1, cfg.max_queries_per_run)

    llm = _make_llm(cfg.query_generator_model)
    plan = _coerce_research_plan(state.get("plan"))

    if not plan or not plan.sections:
        result = _invoke_with_provider_backoff(
            lambda: llm.with_structured_output(SearchQueryList).invoke(
                QUERY_WRITER_PROMPT.format(
                    current_date=_current_date(),
                    research_topic=_get_research_topic(state["messages"]),
                    number_queries=count,
                )
            ),
            attempts=cfg.provider_retry_attempts,
            backoff_seconds=cfg.provider_retry_backoff_seconds,
        )
        return {
            "query_list": [
                {"search_query": q, "section_id": "unplanned"}
                for q in result.query[:count][:max_queries_per_run]  # type: ignore[union-attr]
            ]
        }

    query_list: list[dict[str, str]] = []
    topic = _get_research_topic(state["messages"])
    for section in plan.sections:
        key_questions = "\n".join(f"- {q}" for q in section.key_questions) or "- n/a"
        result = _invoke_with_provider_backoff(
            lambda: llm.with_structured_output(SearchQueryList).invoke(
                SECTION_QUERY_WRITER_PROMPT.format(
                    current_date=_current_date(),
                    research_topic=topic,
                    section_id=section.id,
                    section_title=section.title,
                    section_goal=section.goal,
                    section_key_questions=key_questions,
                    number_queries=count,
                )
            ),
            attempts=cfg.provider_retry_attempts,
            backoff_seconds=cfg.provider_retry_backoff_seconds,
        )
        for query in result.query[:count]:  # type: ignore[union-attr]
            query_list.append({"search_query": query, "section_id": section.id})
    return {"query_list": query_list[:max_queries_per_run]}


def continue_to_web_research(state: QueryGenerationState) -> list[Send]:
    return [
        Send(
            "web_research",
            {
                "search_query": item["search_query"],
                "id": i,
                "section_id": item["section_id"],
            },
        )
        for i, item in enumerate(state["query_list"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    cfg = Configuration.from_runnable_config(config)
    try:
        search_id = int(state["id"])
    except (TypeError, ValueError):
        # Fallback: if id is not an int-like string, still produce deterministic
        # markers within this single process.
        search_id = 0

    citation_base = search_id * cfg.max_citations_per_search
    client = _get_genai_client()
    response = _invoke_with_provider_backoff(
        lambda: client.models.generate_content(
            model=cfg.query_generator_model,
            contents=WEB_SEARCHER_PROMPT.format(
                current_date=_current_date(),
                research_topic=state["search_query"],
            ),
            config={"tools": [{"google_search": {}}], "temperature": 0},
        ),
        attempts=cfg.provider_retry_attempts,
        backoff_seconds=cfg.provider_retry_backoff_seconds,
    )

    marker_sources, text_with_citations = _extract_sources(
        response,
        citation_base=citation_base,
        max_citations_per_search=cfg.max_citations_per_search,
    )

    # Build marker -> URL mapping string for the extraction prompt.
    marker_lines: list[str] = []
    normalized_marker_sources: dict[int, list[dict[str, str]]] = {}
    for marker_id, srcs in sorted(marker_sources.items()):
        normalized_sources: list[dict[str, str]] = []
        for src in srcs:
            raw_url = str(src.get("url", "")).strip()
            if not raw_url:
                continue
            normalized_url = canonicalize_url(raw_url)
            normalized_sources.append(
                {
                    "url": normalized_url or raw_url,
                    "title": str(src.get("title", "")).strip(),
                }
            )
        deduped_sources: dict[str, dict[str, str]] = {}
        for src in normalized_sources:
            u = src.get("url", "")
            if u and u not in deduped_sources:
                deduped_sources[u] = src
        normalized_marker_sources[marker_id] = list(deduped_sources.values())
        urls = [
            s.get("url", "")
            for s in normalized_marker_sources[marker_id]
            if s.get("url")
        ]
        if urls:
            marker_lines.append(f"[{marker_id}] -> {urls[0]}")
    marker_url_mapping = "\n".join(marker_lines) or "(no citation markers found)"

    allowed_tiers = _allowed_tier_set(cfg)
    evidence_items: list[dict] = []
    allowed_items: list[dict] = []
    allowed_snippets: list[str] = []
    extraction_event = (
        f"section={state['section_id']}: evidence extraction returned 0 items"
    )
    try:
        llm = _make_llm(cfg.query_generator_model)
        ev_result = _invoke_with_provider_backoff(
            lambda: llm.with_structured_output(EvidenceList).invoke(
                EVIDENCE_EXTRACTION_PROMPT.format(
                    section_id=state["section_id"],
                    search_query=state["search_query"],
                    marker_url_mapping=marker_url_mapping,
                    text=text_with_citations,
                )
            ),
            attempts=cfg.provider_retry_attempts,
            backoff_seconds=cfg.provider_retry_backoff_seconds,
        )
        if ev_result and hasattr(ev_result, "items") and ev_result.items:
            for item in ev_result.items:  # type: ignore[union-attr]
                item.source_url = canonicalize_url(item.source_url)
                quality = score_url(item.source_url)
                if quality.blocked:
                    continue
                item.source_quality_score = quality.score
                item.source_quality_tier = quality.tier
                item.source_quality_reason = quality.reason
                item_dict = item.model_dump()
                evidence_items.append(item_dict)
                if quality.tier in allowed_tiers:
                    allowed_items.append(item_dict)
                    allowed_snippets.append(
                        f"[{quality.tier} {quality.score:.2f}] {item.claim} (source: {item.source_url})"
                    )
        extraction_event = f"section={state['section_id']}: extracted {len(evidence_items)} evidence items"
    except Exception:
        extraction_event = (
            f"section={state['section_id']}: evidence extraction failed after retries"
        )

    run_id = config.get("configurable", {}).get("run_id")
    if run_id and evidence_items and os.getenv("DATABASE_URL"):
        try:
            from sp26_gke.workflows.research_db import ResearchDB

            evidence_objects = [Evidence(**e) for e in evidence_items]
            db = ResearchDB()
            asyncio.run(db.insert_evidence_batch(run_id, evidence_objects))
        except Exception:
            pass

    return {  # type: ignore[typeddict-item]
        "marker_sources": normalized_marker_sources,
        "search_query": [state["search_query"]],
        "web_research_result": [text_with_citations],
        "evidence_extraction_events": [extraction_event],
        "quality_gate_events": [
            f"section={state['section_id']}: reputable={len(allowed_items)}/{len(evidence_items)}"
        ],
        "section_results": {state["section_id"]: [text_with_citations]},
        "section_evidence": {state["section_id"]: evidence_items},
        "section_allowed_results": {state["section_id"]: allowed_snippets},
        "section_allowed_evidence": {state["section_id"]: allowed_items},
        "section_queries": {state["section_id"]: [state["search_query"]]},
        "section_marker_sources": {state["section_id"]: normalized_marker_sources},
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    cfg = Configuration.from_runnable_config(config)
    loop_count = state.get("research_loop_count", 0) + 1
    model = state.get("reasoning_model") or cfg.reflection_model

    llm = _make_llm(model)
    result = _invoke_with_provider_backoff(
        lambda: llm.with_structured_output(Reflection).invoke(
            REFLECTION_PROMPT.format(
                current_date=_current_date(),
                research_topic=_get_research_topic(state["messages"]),
                summaries="\n\n---\n\n".join(state["web_research_result"]),
            )
        ),
        attempts=cfg.provider_retry_attempts,
        backoff_seconds=cfg.provider_retry_backoff_seconds,
    )
    quality_gate_met, gap_sections, total_allowed, total_reputable = (
        _quality_gate_status(state, cfg)
    )
    reputable_shortfall = max(0, cfg.min_reputable_evidence_total - total_reputable)
    quality_gap = (
        ""
        if quality_gate_met
        else (
            "insufficient_reputable_evidence: "
            f"total={total_allowed}/{cfg.min_allowed_evidence_total}, "
            f"reputable={total_reputable}/{cfg.min_reputable_evidence_total}, "
            f"missing_sections={','.join(gap_sections) if gap_sections else 'none'}"
        )
    )
    return {
        "is_sufficient": result.is_sufficient,  # type: ignore[union-attr]
        "knowledge_gap": result.knowledge_gap,  # type: ignore[union-attr]
        "follow_up_queries": result.follow_up_queries,  # type: ignore[union-attr]
        "quality_gate_met": quality_gate_met,
        "quality_gap_sections": gap_sections,
        "quality_gap": quality_gap,
        "reputable_count": total_reputable,
        "reputable_shortfall": reputable_shortfall,
        "research_loop_count": loop_count,
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState, config: RunnableConfig
) -> str | list[Send]:
    cfg = Configuration.from_runnable_config(config)
    max_loops = state.get("max_research_loops") or cfg.max_research_loops
    if (state["is_sufficient"] and state.get("quality_gate_met", False)) or state[
        "research_loop_count"
    ] >= max_loops:  # type: ignore[operator]
        return "finalize_answer"
    gap_sections = state.get("quality_gap_sections", [])
    follow_up_queries = list(state["follow_up_queries"])
    if gap_sections:
        for section_id in gap_sections:
            follow_up_queries.append(
                f"Find reputable sources and concrete evidence for section: {section_id}"
            )
    deduped_follow_ups: list[str] = []
    seen: set[str] = set()
    for query in follow_up_queries:
        normalized = query.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_follow_ups.append(normalized)
    sends: list[Send] = [
        Send(
            "web_research",
            {
                "search_query": q,
                "id": state["number_of_ran_queries"] + i,
                "section_id": "follow_up",
            },
        )
        for i, q in enumerate(deduped_follow_ups)
    ]
    base_idx = len(sends)
    for j, section_id in enumerate(gap_sections):
        sends.append(
            Send(
                "web_research",
                {
                    "search_query": (
                        f"Find reputable sources with concrete evidence for {section_id}"
                    ),
                    "id": state["number_of_ran_queries"] + base_idx + j,
                    "section_id": section_id,
                },
            )
        )
    reputable_shortfall = int(state.get("reputable_shortfall", 0))
    for k in range(reputable_shortfall):
        sends.append(
            Send(
                "web_research",
                {
                    "search_query": (
                        "Find reputable sources (edu/gov/major research orgs) with "
                        "concrete, citable evidence for this research topic"
                    ),
                    "id": state["number_of_ran_queries"]
                    + base_idx
                    + len(gap_sections)
                    + k,
                    "section_id": "follow_up",
                },
            )
        )
    return sends


def finalize_answer(state: OverallState, config: RunnableConfig) -> OverallState:
    cfg = Configuration.from_runnable_config(config)
    llm = ChatGoogleGenerativeAI(
        model=cfg.answer_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    plan = _coerce_research_plan(state.get("plan"))
    section_order = state.get("section_order", [])
    section_index = _build_section_evidence_index(
        state.get("section_allowed_evidence", {})
    )
    section_payload = _build_evidence_first_payload(
        plan=plan,
        section_order=section_order,
        section_evidence_index=section_index,
    )
    mapped_raw = _invoke_with_provider_backoff(
        lambda: llm.with_structured_output(MappedReport).invoke(
            ANSWER_MAPPED_PROMPT.format(
                current_date=_current_date(),
                research_topic=_get_research_topic(state["messages"]),
                section_order=", ".join(section_order),
                summaries=section_payload,
            )
        ),
        attempts=cfg.provider_retry_attempts,
        backoff_seconds=cfg.provider_retry_backoff_seconds,
    )
    mapped_result: MappedReport = MappedReport.model_validate(mapped_raw)

    section_map: dict[str, PlanSection] = {}
    if isinstance(plan, ResearchPlan):
        section_map = {section.id: section for section in plan.sections}
    generated_sections = {
        section.section_id: section for section in mapped_result.sections
    }
    source_title_lookup = _build_source_title_lookup(state.get("marker_sources", {}))

    url_to_marker: dict[str, int] = {}
    marker_sources: dict[int, dict[str, str]] = {}
    paragraph_mappings: list[dict[str, Any]] = []
    unsupported: list[dict[str, str]] = []
    rendered_sections: list[str] = []

    for section_id in section_order:
        section_info = section_map.get(section_id)
        section_title = (
            section_info.title if section_info else section_id.replace("_", " ").title()
        )
        rendered_sections.append(f"## [{section_id}] {section_title}")
        section_out = generated_sections.get(section_id)
        paragraphs = section_out.paragraphs if section_out else []
        if not paragraphs:
            rendered_sections.append("Gaps: insufficient evidence for this section.")
            continue

        for paragraph_idx, paragraph in enumerate(paragraphs, 1):
            valid_ids = [
                ev_id
                for ev_id in paragraph.evidence_ids
                if ev_id in section_index.get(section_id, {})
            ]
            support_urls: list[str] = []
            for ev_id in valid_ids:
                ev_obj = section_index[section_id][ev_id]
                ev_url = str(ev_obj.get("source_url", "")).strip()
                if ev_url:
                    support_urls.append(ev_url)
            support_urls = sorted(set(support_urls))

            markers: list[int] = []
            for url in support_urls:
                marker_id = url_to_marker.get(url)
                if marker_id is None:
                    marker_id = len(url_to_marker) + 1
                    url_to_marker[url] = marker_id
                    source_title = source_title_lookup.get(url, url)
                    marker_sources[marker_id] = {
                        "title": source_title,
                        "url": url,
                    }
                markers.append(marker_id)

            supported = bool(valid_ids)
            paragraph_text = paragraph.text.strip()
            marker_text = " ".join(f"[{marker_id}]" for marker_id in markers)
            rendered = paragraph_text
            if marker_text:
                rendered = f"{rendered} {marker_text}".strip()
            if not supported:
                rendered = (
                    f"{rendered}\n\nUnsupported (no mapped evidence object).".strip()
                )
                unsupported.append(
                    {
                        "section_id": section_id,
                        "paragraph": paragraph_text,
                        "reason": "no_valid_evidence_ids",
                    }
                )
            rendered_sections.append(rendered)
            paragraph_mappings.append(
                {
                    "section_id": section_id,
                    "paragraph_index": paragraph_idx,
                    "evidence_ids": valid_ids,
                    "supported": supported,
                }
            )
        rendered_sections.append("")

    answer_text = "\n\n".join(rendered_sections).strip()
    entries = [
        {
            "marker_id": str(marker_id),
            "title": src["title"],
            "url": src["url"],
        }
        for marker_id, src in sorted(marker_sources.items(), key=lambda item: item[0])
    ]

    return {
        "messages": [AIMessage(content=answer_text)],
        "sources_gathered": entries,
        "paragraph_evidence_mappings": paragraph_mappings,
        "unsupported_paragraphs": unsupported,
    }  # type: ignore[typeddict-item]


# ── Build graph ──────────────────────────────────────────────────────────────


def _build_graph_builder() -> StateGraph:
    builder = StateGraph(OverallState, config_schema=Configuration)  # type: ignore[call-arg]
    builder.add_node("plan_research", plan_research)
    builder.add_node("generate_query", generate_query)
    builder.add_node("web_research", web_research)
    builder.add_node("reflection", reflection)
    builder.add_node("finalize_answer", finalize_answer)
    builder.add_edge(START, "plan_research")
    builder.add_edge("plan_research", "generate_query")
    builder.add_conditional_edges(
        "generate_query", continue_to_web_research, ["web_research"]
    )
    builder.add_edge("web_research", "reflection")
    builder.add_conditional_edges(
        "reflection", evaluate_research, ["web_research", "finalize_answer"]
    )
    builder.add_edge("finalize_answer", END)
    return builder


def _graph_checkpointer_context(persist: bool):
    """Yield a checkpointer context suited for current runtime mode."""
    if persist and PostgresSaver is not None and os.getenv("DATABASE_URL"):
        return PostgresSaver.from_conn_string(os.environ["DATABASE_URL"])
    return nullcontext(InMemorySaver())


def _compile_graph_with_checkpointer(checkpointer: Any) -> CompiledStateGraph:
    return _build_graph_builder().compile(
        name="pro-search-agent", checkpointer=checkpointer
    )


graph = _build_graph_builder().compile(name="pro-search-agent")


# ── CLI ──────────────────────────────────────────────────────────────────────


def run() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sp26_gke.workflows.research_agent",
        description="Run or resume the deep research agent.",
    )
    parser.add_argument("question", nargs="*", help="Research question for a new run.")
    parser.add_argument(
        "--resume-run-id",
        dest="resume_run_id",
        help="Resume an existing run by run_id/thread_id.",
    )
    args = parser.parse_args()
    question = " ".join(args.question).strip()

    is_resume = bool(args.resume_run_id)
    if not is_resume and not question:
        print('Usage: python -m sp26_gke.workflows.research_agent "Your question"')
        print(
            "Or:    python -m sp26_gke.workflows.research_agent --resume-run-id <run_id>"
        )
        return 1

    run_id = args.resume_run_id or str(uuid.uuid4())
    thread_id = run_id
    persist = bool(os.getenv("DATABASE_URL"))
    if is_resume and not persist:
        print("Resume requires DATABASE_URL for durable checkpoints.", file=sys.stderr)
        return 1
    if is_resume and PostgresSaver is None:
        print(
            "Resume requires langgraph-checkpoint-postgres. Install it and retry.",
            file=sys.stderr,
        )
        return 1

    db: Any | None = None
    if persist:
        from sp26_gke.workflows.research_db import ResearchDB

        db = ResearchDB()
        topic = question or f"resume:{run_id}"
        asyncio.run(db.create_run(run_id, topic, thread_id=thread_id))
        mode = "resume" if is_resume else "start"
        print(f"mode={mode} run_id={run_id}  (persisting to CloudSQL)")
        asyncio.run(db.log_event(run_id, event_type=f"run_{mode}"))
    elif is_resume:
        print("Resume requires persistence enabled.", file=sys.stderr)
        return 1

    if question:
        print(f"\nResearching: {question}\n")

    inputs: dict[str, Any] = {
        "messages": [HumanMessage(content=question)],
        "plan": None,
        "section_order": [],
        "section_results": {},
        "section_evidence": {},
        "section_allowed_results": {},
        "section_allowed_evidence": {},
        "section_queries": {},
        "section_marker_sources": {},
        "search_query": [],
        "web_research_result": [],
        "marker_sources": {},
        "evidence_extraction_events": [],
        "quality_gate_events": [],
        "paragraph_evidence_mappings": [],
        "unsupported_paragraphs": [],
        "sources_gathered": [],
        "research_loop_count": 0,
        "initial_search_query_count": 0,
        "max_research_loops": 0,
        "reasoning_model": "",
    }
    graph_input: dict[str, Any] | None = None if is_resume else inputs

    def _trim(text: str, max_len: int = 160) -> str:
        s = " ".join(text.split())
        if len(s) <= max_len:
            return s
        return s[: max_len - 1] + "…"

    def _section_counts(state_values: dict[str, Any], key: str) -> dict[str, int]:
        raw = state_values.get(key, {})
        if not isinstance(raw, dict):
            return {}
        out: dict[str, int] = {}
        for section_id, items in raw.items():
            if isinstance(section_id, str) and isinstance(items, list):
                out[section_id] = len(items)
        return out

    def _print_progress_from_update(node: str, update: dict[str, Any]) -> None:
        match node:
            case "plan_research":
                section_order = update.get("section_order")
                if isinstance(section_order, list) and section_order:
                    print(
                        f"→ plan_research: planned {len(section_order)} sections",
                        flush=True,
                    )
                    for i, section_id in enumerate(section_order, 1):
                        print(f"  - [{i}] {section_id}", flush=True)
                else:
                    print("→ plan_research", flush=True)
            case "generate_query":
                # Not guaranteed to exist in the update chunk, but if present, print it.
                queries = update.get("query_list")
                if isinstance(queries, list):
                    print(
                        f"→ generate_query: generated {len(queries)} queries",
                        flush=True,
                    )
                    by_section: dict[str, int] = {}
                    for item in queries:
                        if isinstance(item, dict):
                            section_id = str(item.get("section_id", "unplanned"))
                            by_section[section_id] = by_section.get(section_id, 0) + 1
                    for section_id, count in sorted(by_section.items()):
                        print(f"  - {section_id}: {count} queries", flush=True)
                else:
                    print("→ generate_query", flush=True)
            case "web_research":
                section_results = update.get("section_results")
                if isinstance(section_results, dict) and section_results:
                    section_id = next(iter(section_results.keys()))
                    print(f"→ web_research: section={section_id}", flush=True)
                q = update.get("search_query")
                if isinstance(q, list) and q:
                    # Reducer appends, so the newest is the last.
                    newest = str(q[-1])
                    print(f'  searching "{newest}"', flush=True)
                elif isinstance(q, str):
                    print(f'  searching "{q}"', flush=True)
                else:
                    print("→ web_research", flush=True)

                marker_sources = update.get("marker_sources")
                if isinstance(marker_sources, dict):
                    print(
                        f"  extracted {len(marker_sources)} citation markers",
                        flush=True,
                    )
                extraction_events = update.get("evidence_extraction_events")
                if isinstance(extraction_events, list):
                    for event in extraction_events:
                        print(f"  {event}", flush=True)
                quality_events = update.get("quality_gate_events")
                if isinstance(quality_events, list):
                    for event in quality_events:
                        print(f"  {event}", flush=True)
            case "reflection":
                is_sufficient = update.get("is_sufficient")
                loop = update.get("research_loop_count")
                ran = update.get("number_of_ran_queries")
                gap = update.get("knowledge_gap")
                gap_txt = _trim(str(gap)) if gap else ""
                parts: list[str] = []
                if is_sufficient is not None:
                    parts.append(f"sufficient={str(is_sufficient).lower()}")
                if loop is not None:
                    parts.append(f"loop={loop}")
                if ran is not None:
                    parts.append(f"ran_queries={ran}")
                header = "→ reflection" + (": " + " ".join(parts) if parts else "")
                print(header, flush=True)
                if gap_txt:
                    print(f'  gap="{gap_txt}"', flush=True)
                quality_gap = update.get("quality_gap")
                if quality_gap:
                    print(f"  {quality_gap}", flush=True)

                follow_ups = update.get("follow_up_queries")
                if isinstance(follow_ups, list) and follow_ups:
                    print(f"  follow_up_queries={len(follow_ups)}", flush=True)
            case "finalize_answer":
                print("→ finalize_answer", flush=True)
            case _:
                print(f"→ {node}", flush=True)

    last_values: dict[str, Any] | None = None

    config: dict[str, Any] = {
        "configurable": {"run_id": run_id, "thread_id": thread_id}
    }
    try:
        with _graph_checkpointer_context(persist) as checkpointer:
            if persist and hasattr(checkpointer, "setup"):
                checkpointer.setup()
            graph_runner: Any = _compile_graph_with_checkpointer(checkpointer)

            stream_iter = graph_runner.stream(
                graph_input,
                config=config,
                stream_mode=["updates", "values"],
            )
            for mode, chunk in stream_iter:
                if mode == "updates" and isinstance(chunk, dict):
                    for node, update in chunk.items():
                        node_name = str(node)
                        if isinstance(update, dict):
                            _print_progress_from_update(node_name, update)
                        else:
                            print(f"→ {node_name}", flush=True)
                        if db is not None:
                            checkpoint_id = f"{run_id}:{node_name}:{time.time_ns()}"
                            asyncio.run(
                                db.mark_run_status(
                                    run_id,
                                    status="running",
                                    current_node=node_name,
                                    last_checkpoint_id=checkpoint_id,
                                )
                            )
                            asyncio.run(
                                db.upsert_checkpoint_meta(
                                    run_id,
                                    checkpoint_id=checkpoint_id,
                                    node_name=node_name,
                                    attempt=1,
                                )
                            )
                            asyncio.run(
                                db.log_event(
                                    run_id,
                                    event_type="node_progress",
                                    node_name=node_name,
                                )
                            )
                elif mode == "values" and isinstance(chunk, dict):
                    last_values = chunk
                    query_counts = _section_counts(chunk, "section_queries")
                    evidence_counts = _section_counts(chunk, "section_results")
                    allowed_counts = _section_counts(chunk, "section_allowed_evidence")
                    if query_counts or evidence_counts or allowed_counts:
                        print("  section progress:", flush=True)
                        section_ids = sorted(
                            set(query_counts)
                            | set(evidence_counts)
                            | set(allowed_counts)
                        )
                        for section_id in section_ids:
                            q_count = query_counts.get(section_id, 0)
                            e_count = evidence_counts.get(section_id, 0)
                            a_count = allowed_counts.get(section_id, 0)
                            print(
                                f"  - {section_id}: queries={q_count} evidence={e_count} reputable={a_count}",
                                flush=True,
                            )

            if last_values is None:
                last_values = graph_runner.invoke(graph_input, config=config)
    except Exception as exc:
        if db is not None:
            asyncio.run(
                db.mark_run_status(
                    run_id,
                    status="failed",
                    last_error=str(exc),
                )
            )
            asyncio.run(
                db.log_event(
                    run_id,
                    event_type="run_failed",
                    payload={"error": str(exc)},
                )
            )
        raise

    messages = last_values.get("messages", [])
    if messages:
        print(messages[-1].content)

    sources = last_values.get("sources_gathered", [])
    if sources:
        # Aggregate the highest quality tier seen for each URL across all
        # section evidence so we can label each printed source.
        section_evidence_map: dict[str, list[dict[str, Any]]] = last_values.get(
            "section_evidence", {}
        )
        url_to_tier: dict[str, tuple[float, str]] = {}
        for items in section_evidence_map.values():
            for ev in items:
                ev_url = ev.get("source_url", "")
                ev_score = float(ev.get("source_quality_score", 0.5))
                ev_tier = str(ev.get("source_quality_tier", "neutral"))
                if not ev_url:
                    continue
                prior = url_to_tier.get(ev_url)
                if prior is None or ev_score > prior[0]:
                    url_to_tier[ev_url] = (ev_score, ev_tier)

        print(f"\n--- Sources ({len(sources)}) ---")
        for s in sources:
            marker_label = s.get("marker_id")
            label = marker_label if marker_label else "?"
            url = s.get("url", "")
            tier_info = url_to_tier.get(url)
            tier_label = f" ({tier_info[1]} {tier_info[0]:.2f})" if tier_info else ""
            print(f"[{label}]{tier_label} {s.get('title', 'No title')}\n    {url}")

    if persist:
        if db is None:
            raise RuntimeError("Persistence is enabled but ResearchDB is unavailable.")
        asyncio.run(db.complete_run(run_id))
        asyncio.run(db.log_event(run_id, event_type="run_completed"))
        print(f"\n✓ Run {run_id} persisted to CloudSQL.")

    return 0


if __name__ == "__main__":
    sys.exit(run())
