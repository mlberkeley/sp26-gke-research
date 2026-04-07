"""
Deep Research Agent — LangGraph + Gemini with grounded web search.

Based on: https://towardsdatascience.com/langgraph-101-lets-build-a-deep-research-agent/
"""

from __future__ import annotations

import asyncio
import operator
import os
import re
import sys
import threading
import uuid
from datetime import datetime
from typing import Annotated, Any, TypedDict

import google.genai as genai
import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from pydantic import BaseModel, Field, field_validator

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────


class Configuration(BaseModel):
    query_generator_model: str = Field(default="gemini-2.5-flash")
    reflection_model: str = Field(default="gemini-2.5-flash")
    answer_model: str = Field(default="gemini-2.5-flash")
    number_of_initial_queries: int = Field(default=1)
    max_research_loops: int = Field(default=1)
    max_citations_per_search: int = Field(default=5)

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
    section_queries: Annotated[dict[str, list[str]], merge_section_lists]
    section_marker_sources: Annotated[
        dict[str, dict[int, list[dict[str, str]]]],
        merge_section_marker_sources,
    ]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    marker_sources: Annotated[dict[int, list[dict[str, str]]], merge_marker_sources]
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
            r = client.head(url, follow_redirects=True, timeout=5.0)
            return str(r.url)
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


def plan_research(state: OverallState, config: RunnableConfig) -> OverallState:
    cfg = Configuration.from_runnable_config(config)
    llm = _make_llm(cfg.query_generator_model)
    result = llm.with_structured_output(ResearchPlan).invoke(
        PLANNER_PROMPT.format(
            current_date=_current_date(),
            research_topic=_get_research_topic(state["messages"]),
        )
    )
    plan = ResearchPlan.model_validate(result)
    plan_brief = _plan_to_brief(plan)
    return {  # type: ignore[typeddict-item]
        "messages": [AIMessage(content=plan_brief)],
        "plan": plan,
        "section_order": [section.id for section in plan.sections],
    }


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    cfg = Configuration.from_runnable_config(config)
    count = state.get("initial_search_query_count") or cfg.number_of_initial_queries

    llm = _make_llm(cfg.query_generator_model)
    plan_obj = state.get("plan")
    plan = ResearchPlan.model_validate(plan_obj) if plan_obj else None

    if not plan or not plan.sections:
        result = llm.with_structured_output(SearchQueryList).invoke(
            QUERY_WRITER_PROMPT.format(
                current_date=_current_date(),
                research_topic=_get_research_topic(state["messages"]),
                number_queries=count,
            )
        )
        return {
            "query_list": [
                {"search_query": q, "section_id": "unplanned"}
                for q in result.query  # type: ignore[union-attr]
            ]
        }

    query_list: list[dict[str, str]] = []
    topic = _get_research_topic(state["messages"])
    for section in plan.sections:
        key_questions = "\n".join(f"- {q}" for q in section.key_questions) or "- n/a"
        result = llm.with_structured_output(SearchQueryList).invoke(
            SECTION_QUERY_WRITER_PROMPT.format(
                current_date=_current_date(),
                research_topic=topic,
                section_id=section.id,
                section_title=section.title,
                section_goal=section.goal,
                section_key_questions=key_questions,
                number_queries=count,
            )
        )
        for query in result.query:  # type: ignore[union-attr]
            query_list.append({"search_query": query, "section_id": section.id})
    return {"query_list": query_list}


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
    response = client.models.generate_content(
        model=cfg.query_generator_model,
        contents=WEB_SEARCHER_PROMPT.format(
            current_date=_current_date(),
            research_topic=state["search_query"],
        ),
        config={"tools": [{"google_search": {}}], "temperature": 0},
    )

    marker_sources, text_with_citations = _extract_sources(
        response,
        citation_base=citation_base,
        max_citations_per_search=cfg.max_citations_per_search,
    )

    # Build marker -> URL mapping string for the extraction prompt.
    marker_lines: list[str] = []
    for marker_id, srcs in sorted(marker_sources.items()):
        urls = [s.get("url", "") for s in srcs if s.get("url")]
        if urls:
            marker_lines.append(f"[{marker_id}] -> {urls[0]}")
    marker_url_mapping = "\n".join(marker_lines) or "(no citation markers found)"

    evidence_items: list[dict] = []
    try:
        llm = _make_llm(cfg.query_generator_model)
        ev_result = llm.with_structured_output(EvidenceList).invoke(
            EVIDENCE_EXTRACTION_PROMPT.format(
                section_id=state["section_id"],
                search_query=state["search_query"],
                marker_url_mapping=marker_url_mapping,
                text=text_with_citations,
            )
        )
        if ev_result and hasattr(ev_result, "items") and ev_result.items:
            for item in ev_result.items:  # type: ignore[union-attr]
                item.source_url = canonicalize_url(item.source_url)
                evidence_items.append(item.model_dump())
    except Exception:
        pass

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
        "marker_sources": marker_sources,
        "search_query": [state["search_query"]],
        "web_research_result": [text_with_citations],
        "section_results": {state["section_id"]: [text_with_citations]},
        "section_evidence": {state["section_id"]: evidence_items},
        "section_queries": {state["section_id"]: [state["search_query"]]},
        "section_marker_sources": {state["section_id"]: marker_sources},
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    cfg = Configuration.from_runnable_config(config)
    loop_count = state.get("research_loop_count", 0) + 1
    model = state.get("reasoning_model") or cfg.reflection_model

    llm = _make_llm(model)
    result = llm.with_structured_output(Reflection).invoke(
        REFLECTION_PROMPT.format(
            current_date=_current_date(),
            research_topic=_get_research_topic(state["messages"]),
            summaries="\n\n---\n\n".join(state["web_research_result"]),
        )
    )
    return {
        "is_sufficient": result.is_sufficient,  # type: ignore[union-attr]
        "knowledge_gap": result.knowledge_gap,  # type: ignore[union-attr]
        "follow_up_queries": result.follow_up_queries,  # type: ignore[union-attr]
        "research_loop_count": loop_count,
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState, config: RunnableConfig
) -> str | list[Send]:
    cfg = Configuration.from_runnable_config(config)
    max_loops = state.get("max_research_loops") or cfg.max_research_loops
    if state["is_sufficient"] or state["research_loop_count"] >= max_loops:  # type: ignore[operator]
        return "finalize_answer"
    return [
        Send(
            "web_research",
            {
                "search_query": q,
                "id": state["number_of_ran_queries"] + i,
                "section_id": "follow_up",
            },
        )
        for i, q in enumerate(state["follow_up_queries"])
    ]


def finalize_answer(state: OverallState, config: RunnableConfig) -> OverallState:
    cfg = Configuration.from_runnable_config(config)
    llm = ChatGoogleGenerativeAI(
        model=cfg.answer_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    plan_obj = state.get("plan")
    plan = ResearchPlan.model_validate(plan_obj) if plan_obj else None
    section_payload = _build_section_synthesis_payload(
        plan=plan,
        section_order=state.get("section_order", []),
        section_results=state.get("section_results", {}),
        fallback_summaries=state["web_research_result"],
    )
    result = llm.invoke(
        ANSWER_PROMPT.format(
            current_date=_current_date(),
            research_topic=_get_research_topic(state["messages"]),
            section_order=", ".join(state.get("section_order", [])),
            summaries=section_payload,
        )
    )

    answer_text = _content_to_text(result.content)
    marker_id_strs = re.findall(r"\[(\d+)\]", answer_text)
    marker_ids = sorted({int(mid) for mid in marker_id_strs if mid})

    marker_sources: dict[int, list[dict[str, str]]] = state.get("marker_sources", {})
    entries: list[dict[str, str]] = []
    seen_entries: set[tuple[int, str]] = set()

    for marker_id in marker_ids:
        for src in marker_sources.get(marker_id, []):
            url = src.get("url", "")
            if not url:
                continue
            key = (marker_id, url)
            if key in seen_entries:
                continue
            seen_entries.add(key)
            entries.append(
                {
                    "marker_id": str(marker_id),
                    "title": src.get("title", "No title"),
                    "url": url,
                }
            )

    return {
        "messages": [AIMessage(content=answer_text)],
        "sources_gathered": entries,
    }  # type: ignore[typeddict-item]


# ── Build graph ──────────────────────────────────────────────────────────────

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
graph = builder.compile(name="pro-search-agent")


# ── CLI ──────────────────────────────────────────────────────────────────────


def run() -> int:
    if len(sys.argv) < 2:
        print('Usage: python -m sp26_gke.workflows.research_agent "Your question"')
        return 1

    question = " ".join(sys.argv[1:])
    run_id = str(uuid.uuid4())
    persist = bool(os.getenv("DATABASE_URL"))

    if persist:
        from sp26_gke.workflows.research_db import ResearchDB

        db = ResearchDB()
        asyncio.run(db.create_run(run_id, question))
        print(f"run_id={run_id}  (persisting to CloudSQL)")

    print(f"\nResearching: {question}\n")

    inputs: dict[str, Any] = {
        "messages": [HumanMessage(content=question)],
        "plan": None,
        "section_order": [],
        "section_results": {},
        "section_evidence": {},
        "section_queries": {},
        "section_marker_sources": {},
        "search_query": [],
        "web_research_result": [],
        "marker_sources": {},
        "sources_gathered": [],
        "research_loop_count": 0,
        "initial_search_query_count": 0,
        "max_research_loops": 0,
        "reasoning_model": "",
    }

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

                follow_ups = update.get("follow_up_queries")
                if isinstance(follow_ups, list) and follow_ups:
                    print(f"  follow_up_queries={len(follow_ups)}", flush=True)
            case "finalize_answer":
                print("→ finalize_answer", flush=True)
            case _:
                print(f"→ {node}", flush=True)

    last_values: dict[str, Any] | None = None

    # Stream both per-node updates (for progress) and full values (for final output).
    graph_runner: Any = graph
    config: dict[str, Any] = {"configurable": {"run_id": run_id}}
    try:
        stream_iter = graph_runner.stream(
            inputs,
            config=config,
            stream_mode=["updates", "values"],
        )
        for mode, chunk in stream_iter:
            if mode == "updates" and isinstance(chunk, dict):
                for node, update in chunk.items():
                    if isinstance(update, dict):
                        _print_progress_from_update(str(node), update)
                    else:
                        print(f"→ {node}", flush=True)
            elif mode == "values" and isinstance(chunk, dict):
                last_values = chunk
                query_counts = _section_counts(chunk, "section_queries")
                evidence_counts = _section_counts(chunk, "section_results")
                if query_counts or evidence_counts:
                    print("  section progress:", flush=True)
                    section_ids = sorted(set(query_counts) | set(evidence_counts))
                    for section_id in section_ids:
                        q_count = query_counts.get(section_id, 0)
                        e_count = evidence_counts.get(section_id, 0)
                        print(
                            f"  - {section_id}: queries={q_count} evidence={e_count}",
                            flush=True,
                        )
    except Exception:
        # Fallback to updates-only streaming (older LangGraph versions), and use invoke for
        # final output if we can't capture final values.
        last_values = None
        for chunk in graph_runner.stream(inputs, config=config, stream_mode="updates"):
            if not isinstance(chunk, dict):
                continue
            for node, update in chunk.items():
                if isinstance(update, dict):
                    _print_progress_from_update(str(node), update)
                else:
                    print(f"→ {node}", flush=True)

    if last_values is None:
        # If we couldn't capture final state via streaming, do a normal run to get it.
        last_values = graph_runner.invoke(inputs, config=config)

    messages = last_values.get("messages", [])
    if messages:
        print(messages[-1].content)

    sources = last_values.get("sources_gathered", [])
    if sources:
        print(f"\n--- Sources ({len(sources)}) ---")
        for s in sources:
            marker_label = s.get("marker_id")
            label = marker_label if marker_label else "?"
            print(f"[{label}] {s.get('title', 'No title')}\n    {s.get('url', '')}")

    if persist:
        asyncio.run(db.complete_run(run_id))
        print(f"\n✓ Run {run_id} persisted to CloudSQL.")

    return 0


if __name__ == "__main__":
    sys.exit(run())
