"""
Streamlit demo UI for the Deep Research Agent.

Run with: pixi run demo
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any

import streamlit as st
from langchain_core.messages import HumanMessage

from sp26_gke.workflows.research_agent import (
    _compile_graph_with_checkpointer,
    _graph_checkpointer_context,
)

st.set_page_config(page_title="Deep Research Agent", layout="wide")
st.title("Deep Research Agent")


def _trim(text: str, max_len: int = 160) -> str:
    compact_text = " ".join(text.split())
    if len(compact_text) <= max_len:
        return compact_text
    return compact_text[: max_len - 1] + "…"


def _snapshot_node_state(node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Create a compact, display-ready state snapshot for a workflow node."""
    state_snapshot: dict[str, Any] = {
        "node": node_id,
        "recorded_at": time.strftime("%H:%M:%S"),
        "status": "running",
        "summary": "update received",
    }

    if node_id == "plan_research":
        planned_sections = payload.get("section_order")
        section_count = (
            len(planned_sections) if isinstance(planned_sections, list) else 0
        )
        state_snapshot["summary"] = f"planned {section_count} sections"
        state_snapshot["section_count"] = section_count

    elif node_id == "generate_query":
        generated_queries = payload.get("query_list")
        query_count = (
            len(generated_queries) if isinstance(generated_queries, list) else 0
        )
        state_snapshot["summary"] = f"generated {query_count} queries"
        state_snapshot["query_count"] = query_count

    elif node_id == "web_research":
        current_section_id = ""
        node_section_results = payload.get("section_results")
        if isinstance(node_section_results, dict) and node_section_results:
            current_section_id = str(next(iter(node_section_results.keys())))
        node_marker_sources = payload.get("marker_sources")
        marker_count = (
            len(node_marker_sources) if isinstance(node_marker_sources, dict) else 0
        )
        summary_parts: list[str] = []
        if current_section_id:
            summary_parts.append(f"section={current_section_id}")
            state_snapshot["section_id"] = current_section_id
        summary_parts.append(f"markers={marker_count}")
        state_snapshot["summary"] = " ".join(summary_parts)
        state_snapshot["marker_count"] = marker_count

    elif node_id == "reflection":
        reflection_sufficient = payload.get("is_sufficient")
        reflection_loop = payload.get("research_loop_count")
        summary_parts = []
        if reflection_sufficient is not None:
            summary_parts.append(f"sufficient={str(reflection_sufficient).lower()}")
            state_snapshot["is_sufficient"] = reflection_sufficient
            if reflection_sufficient:
                state_snapshot["status"] = "complete"
        if reflection_loop is not None:
            summary_parts.append(f"loop={reflection_loop}")
            state_snapshot["loop"] = reflection_loop
        reflection_gap = payload.get("knowledge_gap")
        if reflection_gap:
            summary_parts.append(f'gap="{_trim(str(reflection_gap), max_len=80)}"')
        state_snapshot["summary"] = (
            " ".join(summary_parts) if summary_parts else "reflection update"
        )

    elif node_id == "finalize_answer":
        state_snapshot["status"] = "complete"
        state_snapshot["summary"] = "final report generated"

    return state_snapshot


def _ordered_nodes(latest_by_node: dict[str, dict[str, Any]]) -> list[str]:
    """Sort known workflow nodes first and keep unknown nodes after."""
    workflow_order = [
        "plan_research",
        "generate_query",
        "web_research",
        "reflection",
        "finalize_answer",
    ]
    known = [name for name in workflow_order if name in latest_by_node]
    unknown = sorted([name for name in latest_by_node if name not in workflow_order])
    return known + unknown


topic = st.text_input("Research topic", placeholder="e.g. state of LPU hardware")
run_button = st.button("Run", type="primary")

if run_button and topic:
    run_id = str(uuid.uuid4())
    database_url = os.getenv("DATABASE_URL")
    persist = bool(database_url and database_url.strip())
    db: Any | None = None

    inputs: dict[str, Any] = {
        "messages": [HumanMessage(content=topic)],
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
    config: dict[str, Any] = {"configurable": {"run_id": run_id, "thread_id": run_id}}

    progress_container = st.container()
    report_placeholder = st.empty()
    sources_placeholder = st.empty()
    evidence_placeholder = st.empty()
    node_states_placeholder = st.empty()
    db_placeholder = st.empty()
    checkpoints_placeholder = st.empty()

    last_values: dict[str, Any] | None = None
    node_latest_state: dict[str, dict[str, Any]] = {}
    node_timeline: list[dict[str, Any]] = []

    if persist:
        from sp26_gke.workflows.research_db import ResearchDB

        db = ResearchDB(dsn=database_url)
        asyncio.run(db.create_run(run_id, topic, thread_id=run_id))
        asyncio.run(db.log_event(run_id, event_type="run_start"))

    with progress_container:
        status = st.status("Researching...", expanded=True)

        with _graph_checkpointer_context(persist=persist) as saver:
            if persist and hasattr(saver, "setup"):
                saver.setup()
            graph_runner: Any = _compile_graph_with_checkpointer(saver)
            try:
                stream_iter = graph_runner.stream(
                    inputs,
                    config=config,
                    stream_mode=["updates", "values"],
                )
                for mode, chunk in stream_iter:
                    if mode == "updates" and isinstance(chunk, dict):
                        for node, update in chunk.items():
                            node_name = str(node)
                            if not isinstance(update, dict):
                                status.write(f"→ {node_name}")
                                continue

                            if node_name == "plan_research":
                                section_order = update.get("section_order")
                                if isinstance(section_order, list) and section_order:
                                    line = f"→ plan_research: planned {len(section_order)} sections"
                                    status.write(line)
                                    for i, sid in enumerate(section_order, 1):
                                        status.write(f"  - [{i}] {sid}")
                                else:
                                    status.write("→ plan_research")

                            elif node_name == "generate_query":
                                queries = update.get("query_list")
                                if isinstance(queries, list):
                                    line = f"→ generate_query: generated {len(queries)} queries"
                                    status.write(line)
                                    by_section: dict[str, int] = {}
                                    for item in queries:
                                        if isinstance(item, dict):
                                            sid = str(
                                                item.get("section_id", "unplanned")
                                            )
                                            by_section[sid] = by_section.get(sid, 0) + 1
                                    for sid, count in sorted(by_section.items()):
                                        status.write(f"  - {sid}: {count} queries")
                                else:
                                    status.write("→ generate_query")

                            elif node_name == "web_research":
                                section_results = update.get("section_results")
                                if (
                                    isinstance(section_results, dict)
                                    and section_results
                                ):
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
                                extraction_events = update.get(
                                    "evidence_extraction_events"
                                )
                                if isinstance(extraction_events, list):
                                    for event in extraction_events:
                                        status.write(f"  {event}")
                                quality_events = update.get("quality_gate_events")
                                if isinstance(quality_events, list):
                                    for event in quality_events:
                                        status.write(f"  {event}")

                            elif node_name == "reflection":
                                is_sufficient = update.get("is_sufficient")
                                loop = update.get("research_loop_count")
                                gap = update.get("knowledge_gap")
                                gap_txt = _trim(str(gap)) if gap else ""
                                parts: list[str] = []
                                if is_sufficient is not None:
                                    parts.append(
                                        f"sufficient={str(is_sufficient).lower()}"
                                    )
                                if loop is not None:
                                    parts.append(f"loop={loop}")
                                header = "→ reflection" + (
                                    ": " + " ".join(parts) if parts else ""
                                )
                                status.write(header)
                                if gap_txt:
                                    status.write(f'  gap="{gap_txt}"')
                                quality_gap = update.get("quality_gap")
                                if quality_gap:
                                    status.write(f"  {quality_gap}")

                            elif node_name == "finalize_answer":
                                status.write("→ Generating final report...")

                            else:
                                status.write(f"→ {node_name}")

                            snapshot = _snapshot_node_state(node_name, update)
                            node_latest_state[node_name] = snapshot
                            node_timeline.append(snapshot)

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

            except Exception:
                last_values = None
                for chunk in graph_runner.stream(
                    inputs, config=config, stream_mode="updates"
                ):
                    if not isinstance(chunk, dict):
                        continue
                    for node, _update in chunk.items():
                        node_name = str(node)
                        status.write(f"→ {node_name}")
                        fallback_snapshot = {
                            "node": node_name,
                            "recorded_at": time.strftime("%H:%M:%S"),
                            "status": "running",
                            "summary": "update received",
                        }
                        node_latest_state[node_name] = fallback_snapshot
                        node_timeline.append(fallback_snapshot)

            if last_values is None:
                last_values = graph_runner.invoke(inputs, config=config)

        status.update(label="Research complete", state="complete", expanded=False)

    if db is not None:
        asyncio.run(db.complete_run(run_id))
        asyncio.run(db.log_event(run_id, event_type="run_completed"))

    messages = last_values.get("messages", [])
    if messages:
        report_placeholder.markdown(messages[-1].content)

    sources = last_values.get("sources_gathered", [])
    if sources:
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
        with sources_placeholder.expander(f"Sources ({len(sources)})", expanded=False):
            for s in sources:
                marker_label = s.get("marker_id", "?")
                title = s.get("title", "No title")
                url = s.get("url", "")
                tier_info = url_to_tier.get(url)
                tier_label = (
                    f" ({tier_info[1]} {tier_info[0]:.2f})" if tier_info else ""
                )
                st.markdown(f"**[{marker_label}]**{tier_label} {title}  \n{url}")

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
                        tier = ev.get("source_quality_tier", "neutral")
                        score = float(ev.get("source_quality_score", 0.5))
                        st.markdown(
                            f"- `[{tier} {score:.2f}]` {claim}  \n  [source]({url})"
                        )

    unsupported_paragraphs: list[dict[str, str]] = last_values.get(
        "unsupported_paragraphs", []
    )
    if unsupported_paragraphs:
        with evidence_placeholder.container():
            st.caption(f"Unsupported paragraphs flagged: {len(unsupported_paragraphs)}")

    if node_latest_state:
        with node_states_placeholder.container():
            st.subheader("Node States")
            st.caption("Latest state for each workflow node")

            latest_rows: list[dict[str, Any]] = []
            for node_name in _ordered_nodes(node_latest_state):
                snapshot = node_latest_state[node_name]
                latest_rows.append(
                    {
                        "node": node_name,
                        "status": snapshot.get("status", "running"),
                        "updated_at": snapshot.get("recorded_at", ""),
                        "summary": snapshot.get("summary", ""),
                    }
                )
            st.table(latest_rows)

            with st.expander(
                f"Execution Timeline ({len(node_timeline)} updates)", expanded=False
            ):
                for idx, snapshot in enumerate(node_timeline, start=1):
                    st.markdown(
                        f"{idx}. `{snapshot.get('recorded_at', '')}` "
                        f"`{snapshot.get('node', '')}` "
                        f"{snapshot.get('summary', '')}"
                    )

    if persist:
        try:
            summary = (
                asyncio.run(db.get_run_persistence_summary(run_id))
                if db is not None
                else {}
            )
            with db_placeholder.container():
                st.subheader("DB Persistence Verification")
                run_info = summary.get("run") or {}
                st.write(f"run_id: `{run_id}`")
                st.write(f"status: `{run_info.get('status', 'unknown')}`")
                st.write(f"thread_id: `{run_info.get('thread_id', 'n/a')}`")
                st.write(
                    f"evidence_rows: `{summary.get('evidence_count', 0)}` | "
                    f"event_rows: `{summary.get('event_count', 0)}`"
                )
            if db is not None:
                checkpoints = asyncio.run(
                    db.get_run_checkpoint_timeline(run_id, limit=30)
                )
                with checkpoints_placeholder.container():
                    st.subheader("Checkpoint Timeline")
                    if not checkpoints:
                        st.caption("No checkpoint rows found for this run yet.")
                    else:
                        for row in checkpoints:
                            node = row.get("node_name") or "(unknown)"
                            checkpoint_id = row.get("checkpoint_id", "")
                            created_at = row.get("created_at")
                            st.markdown(
                                f"- `{created_at}` | `{node}` | `{checkpoint_id}`"
                            )
        except Exception as exc:
            with db_placeholder.container():
                st.subheader("DB Persistence Verification")
                st.warning(f"Could not read DB summary: {exc}")
