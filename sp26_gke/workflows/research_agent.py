"""
Deep Research Agent — LangGraph + Gemini with grounded web search.

Based on: https://towardsdatascience.com/langgraph-101-lets-build-a-deep-research-agent/
"""

from __future__ import annotations

import operator
import os
import re
import sys
from datetime import datetime
from typing import Annotated, Any, TypedDict

import google.genai as genai
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from pydantic import BaseModel, Field

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────


class Configuration(BaseModel):
    query_generator_model: str = Field(default="gemini-2.5-flash")
    reflection_model: str = Field(default="gemini-2.5-flash")
    answer_model: str = Field(default="gemini-2.5-flash")
    number_of_initial_queries: int = Field(default=3)
    max_research_loops: int = Field(default=1)
    max_citations_per_search: int = Field(default=20)

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


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    marker_sources: Annotated[dict[int, list[dict[str, str]]], merge_marker_sources]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str


class QueryGenerationState(TypedDict):
    query_list: list


class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    research_loop_count: int
    number_of_ran_queries: int


class WebSearchState(TypedDict):
    search_query: str
    id: str


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


QUERY_WRITER_PROMPT = """Generate {number_queries} diverse, targeted web search queries to research this topic.
Current date: {current_date}
Topic: {research_topic}

Rules:
- Prefer a single query unless the topic has multiple distinct aspects
- Queries must be specific and likely to return current, authoritative results
- No duplicate or near-duplicate queries

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

ANSWER_PROMPT = """Synthesise the following research into a comprehensive, well-structured answer.
Current date: {current_date}
Topic: {research_topic}

Research:
{summaries}

Write a clear, thorough answer with inline citations (e.g. [1], [2]) where relevant."""


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
    max_citations_per_search: int = 50,
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


# ── Nodes ────────────────────────────────────────────────────────────────────

_genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def _make_llm(model: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    cfg = Configuration.from_runnable_config(config)
    count = state.get("initial_search_query_count") or cfg.number_of_initial_queries

    llm = _make_llm(cfg.query_generator_model)
    result = llm.with_structured_output(SearchQueryList).invoke(
        QUERY_WRITER_PROMPT.format(
            current_date=_current_date(),
            research_topic=_get_research_topic(state["messages"]),
            number_queries=count,
        )
    )
    return {"query_list": result.query}  # type: ignore[union-attr]


def continue_to_web_research(state: QueryGenerationState) -> list[Send]:
    return [
        Send("web_research", {"search_query": q, "id": i})
        for i, q in enumerate(state["query_list"])
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
    response = _genai_client.models.generate_content(
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
    return {  # type: ignore[typeddict-item]
        "marker_sources": marker_sources,
        "search_query": [state["search_query"]],
        "web_research_result": [text_with_citations],
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
            {"search_query": q, "id": state["number_of_ran_queries"] + i},
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
    result = llm.invoke(
        ANSWER_PROMPT.format(
            current_date=_current_date(),
            research_topic=_get_research_topic(state["messages"]),
            summaries="\n\n---\n\n".join(state["web_research_result"]),
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
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)
builder.add_edge(START, "generate_query")
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
    print(f"\nResearching: {question}\n")

    inputs: dict[str, Any] = {
        "messages": [HumanMessage(content=question)],
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

    def _print_progress_from_update(node: str, update: dict[str, Any]) -> None:
        match node:
            case "generate_query":
                # Not guaranteed to exist in the update chunk, but if present, print it.
                queries = update.get("query_list")
                if isinstance(queries, list):
                    print(
                        f"→ generate_query: generated {len(queries)} queries",
                        flush=True,
                    )
                    for i, q in enumerate(queries):
                        print(f"  - [{i}] {q}", flush=True)
                else:
                    print("→ generate_query", flush=True)
            case "web_research":
                q = update.get("search_query")
                if isinstance(q, list) and q:
                    # Reducer appends, so the newest is the last.
                    newest = str(q[-1])
                    print(f'→ web_research: searching "{newest}"', flush=True)
                elif isinstance(q, str):
                    print(f'→ web_research: searching "{q}"', flush=True)
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
    try:
        stream_iter = graph_runner.stream(
            inputs,
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
    except Exception:
        # Fallback to updates-only streaming (older LangGraph versions), and use invoke for
        # final output if we can't capture final values.
        for chunk in graph_runner.stream(inputs, stream_mode="updates"):
            if not isinstance(chunk, dict):
                continue
            for node, update in chunk.items():
                if isinstance(update, dict):
                    _print_progress_from_update(str(node), update)
                else:
                    print(f"→ {node}", flush=True)

    if last_values is None:
        # If we couldn't capture final state via streaming, do a normal run to get it.
        last_values = graph_runner.invoke(inputs)

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

    return 0


if __name__ == "__main__":
    sys.exit(run())
