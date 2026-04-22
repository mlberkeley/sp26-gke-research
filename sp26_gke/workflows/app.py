"""
Streamlit demo UI for the Deep Research Agent.

Run with: pixi run demo
"""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st
from langchain_core.messages import HumanMessage

from sp26_gke.workflows.research_agent import graph as _graph

st.set_page_config(page_title="Deep Research Agent", layout="wide")
st.title("Deep Research Agent")

# Cast to Any so mypy doesn't complain about stream/invoke overload signatures.
graph_runner: Any = _graph


def _trim(text: str, max_len: int = 160) -> str:
    s = " ".join(text.split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


topic = st.text_input("Research topic", placeholder="e.g. state of LPU hardware")
run_button = st.button("Run", type="primary")

if run_button and topic:
    run_id = str(uuid.uuid4())

    inputs: dict[str, Any] = {
        "messages": [HumanMessage(content=topic)],
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
    config: dict[str, Any] = {"configurable": {"run_id": run_id}}

    progress_container = st.container()
    report_placeholder = st.empty()
    sources_placeholder = st.empty()
    evidence_placeholder = st.empty()

    last_values: dict[str, Any] | None = None

    with progress_container:
        status = st.status("Researching...", expanded=True)

        try:
            stream_iter = graph_runner.stream(
                inputs,
                config=config,
                stream_mode=["updates", "values"],
            )
            for mode, chunk in stream_iter:
                if mode == "updates" and isinstance(chunk, dict):
                    for node, update in chunk.items():
                        if not isinstance(update, dict):
                            status.write(f"→ {node}")
                            continue

                        if node == "plan_research":
                            section_order = update.get("section_order")
                            if isinstance(section_order, list) and section_order:
                                line = f"→ plan_research: planned {len(section_order)} sections"
                                status.write(line)
                                for i, sid in enumerate(section_order, 1):
                                    status.write(f"  - [{i}] {sid}")
                            else:
                                status.write("→ plan_research")

                        elif node == "generate_query":
                            queries = update.get("query_list")
                            if isinstance(queries, list):
                                line = f"→ generate_query: generated {len(queries)} queries"
                                status.write(line)
                                by_section: dict[str, int] = {}
                                for item in queries:
                                    if isinstance(item, dict):
                                        sid = str(item.get("section_id", "unplanned"))
                                        by_section[sid] = by_section.get(sid, 0) + 1
                                for sid, count in sorted(by_section.items()):
                                    status.write(f"  - {sid}: {count} queries")
                            else:
                                status.write("→ generate_query")

                        elif node == "web_research":
                            section_results = update.get("section_results")
                            if isinstance(section_results, dict) and section_results:
                                sid = next(iter(section_results.keys()))
                                status.write(f"→ web_research: section={sid}")
                            q = update.get("search_query")
                            if isinstance(q, list) and q:
                                status.write(f'  searching "{q[-1]}"')
                            marker_sources = update.get("marker_sources")
                            if isinstance(marker_sources, dict):
                                status.write(
                                    f"  extracted {len(marker_sources)} citation markers"
                                )

                        elif node == "reflection":
                            is_sufficient = update.get("is_sufficient")
                            loop = update.get("research_loop_count")
                            gap = update.get("knowledge_gap")
                            gap_txt = _trim(str(gap)) if gap else ""
                            parts: list[str] = []
                            if is_sufficient is not None:
                                parts.append(f"sufficient={str(is_sufficient).lower()}")
                            if loop is not None:
                                parts.append(f"loop={loop}")
                            header = "→ reflection" + (
                                ": " + " ".join(parts) if parts else ""
                            )
                            status.write(header)
                            if gap_txt:
                                status.write(f'  gap="{gap_txt}"')

                        elif node == "finalize_answer":
                            status.write("→ Generating final report...")

                        else:
                            status.write(f"→ {node}")

                elif mode == "values" and isinstance(chunk, dict):
                    last_values = chunk

        except Exception:
            last_values = None
            for chunk in graph_runner.stream(
                inputs, config=config, stream_mode="updates"
            ):
                if not isinstance(chunk, dict):
                    continue
                for node, _update in chunk.items():
                    status.write(f"→ {node}")

        if last_values is None:
            last_values = graph_runner.invoke(inputs, config=config)

        status.update(label="Research complete", state="complete", expanded=False)

    messages = last_values.get("messages", [])
    if messages:
        report_placeholder.markdown(messages[-1].content)

    sources = last_values.get("sources_gathered", [])
    if sources:
        with sources_placeholder.expander(f"Sources ({len(sources)})", expanded=False):
            for s in sources:
                marker_label = s.get("marker_id", "?")
                title = s.get("title", "No title")
                url = s.get("url", "")
                st.markdown(f"**[{marker_label}]** {title}  \n{url}")

    section_evidence: dict[str, list[dict[str, Any]]] = last_values.get(
        "section_evidence", {}
    )
    if section_evidence:
        with evidence_placeholder.container():
            st.subheader("Structured Evidence")
            for section_id, items in section_evidence.items():
                if not items:
                    continue
                with st.expander(section_id, expanded=False):
                    for ev in items:
                        claim = ev.get("claim", "")
                        url = ev.get("source_url", "")
                        st.markdown(f"- {claim}  \n  [source]({url})")
