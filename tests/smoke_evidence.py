"""Quick smoke test for evidence extraction (no DB required)."""

from typing import Any

from langchain_core.messages import HumanMessage
from sp26_gke.workflows.research_agent import graph

inputs: dict[str, Any] = {
    "messages": [HumanMessage(content="What is an LPU?")],
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
result = graph.invoke(inputs)  # type: ignore[call-overload]

ev = result.get("section_evidence", {})
if not ev:
    print("No evidence extracted.")
else:
    for section_id, items in ev.items():
        print(f"\n{section_id}: {len(items)} evidence items")
        for e in items[:2]:
            print(f"  - {e['claim'][:80]}...")
            print(f"    {e['source_url']}")
