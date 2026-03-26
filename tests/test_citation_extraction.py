import types

from langchain_core.messages import HumanMessage
from sp26_gke.workflows import research_agent as ra


class DummyWeb:
    def __init__(self, uri: str, title: str) -> None:
        self.uri = uri
        self.title = title


class DummyGroundingChunk:
    def __init__(self, web: DummyWeb) -> None:
        self.web = web


class DummySegment:
    def __init__(self, end_index: int) -> None:
        self.end_index = end_index


class DummySupport:
    def __init__(
        self, segment: DummySegment, grounding_chunk_indices: list[int]
    ) -> None:
        self.segment = segment
        self.grounding_chunk_indices = grounding_chunk_indices


class DummyGroundingMetadata:
    def __init__(
        self,
        grounding_chunks: list[DummyGroundingChunk],
        grounding_supports: list[DummySupport],
    ) -> None:
        self.grounding_chunks = grounding_chunks
        self.grounding_supports = grounding_supports


class DummyCandidate:
    def __init__(self, grounding_metadata: DummyGroundingMetadata) -> None:
        self.grounding_metadata = grounding_metadata


class DummyResponse:
    def __init__(self, text: str, grounding_metadata: DummyGroundingMetadata) -> None:
        self.text = text
        self.candidates = [DummyCandidate(grounding_metadata=grounding_metadata)]


def test_extract_sources_marker_ids_and_urls() -> None:
    # Two supports with different `segment.end_index` values.
    # - end_index=7 -> marker id base+1, grounded on chunk 0 + 1 (two URLs)
    # - end_index=3 -> marker id base+2, grounded on chunk 2 (single URL)
    text = "abcdefghij"

    chunks = [
        DummyGroundingChunk(DummyWeb("https://a.com", "A")),
        DummyGroundingChunk(DummyWeb("https://b.com", "B")),
        DummyGroundingChunk(DummyWeb("https://c.com", "C")),
    ]
    supports = [
        DummySupport(DummySegment(end_index=7), grounding_chunk_indices=[0, 1]),
        DummySupport(DummySegment(end_index=3), grounding_chunk_indices=[2]),
    ]

    metadata = DummyGroundingMetadata(
        grounding_chunks=chunks, grounding_supports=supports
    )
    response = DummyResponse(text=text, grounding_metadata=metadata)

    marker_sources, out_text = ra._extract_sources(
        response,
        citation_base=0,
        max_citations_per_search=10,
    )

    assert set(marker_sources.keys()) == {1, 2}
    assert {s["url"] for s in marker_sources[1]} == {"https://a.com", "https://b.com"}
    assert {s["url"] for s in marker_sources[2]} == {"https://c.com"}

    # Insertion happens in descending `end_index` order, so marker ids should appear swapped.
    # First insert at 7: "[1]" then insert at 3: "[2]" -> abc [2] defg [1] hij
    assert out_text == "abc [2]defg [1]hij"


def test_extract_sources_marker_id_collision_avoidance() -> None:
    text = "abcdefghij"
    chunks = [
        DummyGroundingChunk(DummyWeb("https://a.com", "A")),
        DummyGroundingChunk(DummyWeb("https://b.com", "B")),
        DummyGroundingChunk(DummyWeb("https://c.com", "C")),
    ]
    supports = [
        DummySupport(DummySegment(end_index=7), grounding_chunk_indices=[0, 1]),
        DummySupport(DummySegment(end_index=3), grounding_chunk_indices=[2]),
    ]
    metadata = DummyGroundingMetadata(
        grounding_chunks=chunks, grounding_supports=supports
    )
    response = DummyResponse(text=text, grounding_metadata=metadata)

    marker_sources, _ = ra._extract_sources(
        response,
        citation_base=100,
        max_citations_per_search=10,
    )

    assert set(marker_sources.keys()) == {101, 102}


def test_extract_sources_respects_max_citations_per_search() -> None:
    text = "abcdefghij"
    chunks = [
        DummyGroundingChunk(DummyWeb("https://a.com", "A")),
        DummyGroundingChunk(DummyWeb("https://b.com", "B")),
        DummyGroundingChunk(DummyWeb("https://c.com", "C")),
    ]
    supports = [
        DummySupport(DummySegment(end_index=7), grounding_chunk_indices=[0, 1]),
        DummySupport(DummySegment(end_index=3), grounding_chunk_indices=[2]),
    ]
    metadata = DummyGroundingMetadata(
        grounding_chunks=chunks, grounding_supports=supports
    )
    response = DummyResponse(text=text, grounding_metadata=metadata)

    marker_sources, out_text = ra._extract_sources(
        response,
        citation_base=0,
        max_citations_per_search=1,
    )

    assert set(marker_sources.keys()) == {1}
    assert "[1]" in out_text
    assert "[2]" not in out_text


def test_finalize_answer_filters_sources_to_markers_used(monkeypatch) -> None:
    class DummyChat:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def invoke(self, _prompt: str) -> object:
            return types.SimpleNamespace(
                content="Answer with [11] and [12], plus an invented [999]."
            )

    monkeypatch.setattr(ra, "ChatGoogleGenerativeAI", DummyChat)

    state: ra.OverallState = {
        "messages": [HumanMessage(content="Topic")],
        "plan": None,
        "section_order": [],
        "section_results": {},
        "section_queries": {},
        "section_marker_sources": {},
        "section_evidence_items": {},
        "search_query": [],
        "web_research_result": ["evidence with [11] and [12] markers"],
        "marker_sources": {
            11: [{"url": "https://s11.com", "title": "S11"}],
            12: [
                {"url": "https://s12a.com", "title": "S12A"},
                {"url": "https://s12b.com", "title": "S12B"},
            ],
        },
        "sources_gathered": [],
        "initial_search_query_count": 1,
        "max_research_loops": 1,
        "research_loop_count": 0,
        "reasoning_model": "gemini-2.5-flash",
    }

    result = ra.finalize_answer(state, config={})
    entries = result["sources_gathered"]

    assert {e["marker_id"] for e in entries} == {"11", "12"}
    assert len(entries) == 3
    assert any(
        e["marker_id"] == "12" and e["url"] == "https://s12b.com" for e in entries
    )


def test_plan_research_sets_plan_and_section_order(monkeypatch) -> None:
    class DummyLLM:
        def with_structured_output(self, _schema: object) -> "DummyLLM":
            return self

        def invoke(self, _prompt: str) -> ra.ResearchPlan:
            return ra.ResearchPlan(
                topic_rewrite="Clarified topic",
                overall_success_criteria="Enough evidence per section",
                sections=[
                    ra.PlanSection(
                        id="scope_and_definitions",
                        title="Scope and Definitions",
                        goal="Define terms and scope",
                        key_questions=["What is in scope?"],
                        query_hints=["definition"],
                        required=True,
                    ),
                    ra.PlanSection(
                        id="findings",
                        title="Findings",
                        goal="Summarize technical findings",
                        key_questions=["What are the main findings?"],
                        query_hints=["benchmarks"],
                        required=True,
                    ),
                ],
            )

    monkeypatch.setattr(ra, "_make_llm", lambda _model: DummyLLM())

    state: ra.OverallState = {
        "messages": [HumanMessage(content="Topic")],
        "plan": None,
        "section_order": [],
        "section_results": {},
        "section_queries": {},
        "section_marker_sources": {},
        "section_evidence_items": {},
        "search_query": [],
        "web_research_result": [],
        "marker_sources": {},
        "sources_gathered": [],
        "initial_search_query_count": 1,
        "max_research_loops": 1,
        "research_loop_count": 0,
        "reasoning_model": "gemini-2.5-flash",
    }

    result = ra.plan_research(state, config={})
    plan = result["plan"]
    assert plan is not None
    assert [s.id for s in plan.sections] == ["scope_and_definitions", "findings"]
    assert result["section_order"] == ["scope_and_definitions", "findings"]


def test_continue_to_web_research_preserves_section_ids() -> None:
    sends = ra.continue_to_web_research(
        {
            "query_list": [
                {"search_query": "q1", "section_id": "scope_and_definitions"},
                {"search_query": "q2", "section_id": "findings"},
            ]
        }
    )
    assert len(sends) == 2
    assert sends[0].arg["section_id"] == "scope_and_definitions"
    assert sends[1].arg["section_id"] == "findings"


def test_web_research_returns_section_scoped_outputs(monkeypatch) -> None:
    class DummyModels:
        def generate_content(self, **_kwargs: object) -> object:
            return object()

    class DummyClient:
        def __init__(self) -> None:
            self.models = DummyModels()

    monkeypatch.setattr(ra, "_get_genai_client", lambda: DummyClient())
    monkeypatch.setattr(
        ra,
        "_extract_sources",
        lambda *_args, **_kwargs: (
            {21: [{"url": "https://x.com", "title": "X"}]},
            "text [21]",
        ),
    )

    result = ra.web_research(
        {"search_query": "query", "id": "3", "section_id": "findings"},
        config={},
    )

    assert result["section_queries"] == {"findings": ["query"]}
    assert result["section_results"] == {"findings": ["text [21]"]}
    assert "findings" in result["section_marker_sources"]


def test_build_section_synthesis_payload_orders_sections() -> None:
    plan = ra.ResearchPlan(
        topic_rewrite="T",
        overall_success_criteria="C",
        sections=[
            ra.PlanSection(id="b", title="B", goal="Goal B"),
            ra.PlanSection(id="a", title="A", goal="Goal A"),
        ],
    )
    payload = ra._build_section_synthesis_payload(
        plan=plan,
        section_order=["a", "b"],
        section_results={"a": ["A1"], "b": ["B1"]},
        fallback_summaries=[],
    )
    assert payload.index("id: a") < payload.index("id: b")


def test_build_section_synthesis_payload_marks_no_evidence() -> None:
    payload = ra._build_section_synthesis_payload(
        plan=None,
        section_order=["scope_and_definitions"],
        section_results={},
        fallback_summaries=["legacy summary"],
    )
    assert "- (no evidence)" in payload
