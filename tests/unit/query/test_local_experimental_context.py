# Copyright (C) 2026 Microsoft Corporation.
# Licensed under the MIT License

from __future__ import annotations

from graphrag.data_model.community import Community
from graphrag.data_model.community_report import CommunityReport
from graphrag.data_model.covariate import Covariate
from graphrag.data_model.entity import Entity
from graphrag.query.context_builder.conversation_history import ConversationHistory
from graphrag.query.structured_search.local_search.community_selection import (
    select_community_reports,
)
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)


class DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [1] * len(text.split())

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(["x"] * len(token_ids))

    def num_tokens(self, text: str) -> int:
        return len(self.encode(text))


def _make_communities() -> list[Community]:
    return [
        Community(
            id="cid-root",
            short_id="root",
            title="root",
            level="0",
            parent="",
            children=["p1"],
        ),
        Community(
            id="cid-p1",
            short_id="p1",
            title="parent",
            level="1",
            parent="root",
            children=["l1", "l2"],
        ),
        Community(
            id="cid-l1",
            short_id="l1",
            title="leaf1",
            level="2",
            parent="p1",
            children=[],
        ),
        Community(
            id="cid-l2",
            short_id="l2",
            title="leaf2",
            level="2",
            parent="p1",
            children=[],
        ),
    ]


def _make_reports() -> list[CommunityReport]:
    return [
        CommunityReport(
            id="rid-l1",
            short_id="l1",
            title="leaf1",
            community_id="l1",
            summary="summary l1",
            full_content="full content l1 forbidden",
            rank=9.0,
        ),
        CommunityReport(
            id="rid-l2",
            short_id="l2",
            title="leaf2",
            community_id="l2",
            summary="summary l2",
            full_content="full content l2 forbidden",
            rank=8.0,
        ),
        CommunityReport(
            id="rid-p1",
            short_id="p1",
            title="parent",
            community_id="p1",
            summary="summary p1",
            full_content="full content p1 forbidden",
            rank=7.0,
        ),
        CommunityReport(
            id="rid-root",
            short_id="root",
            title="root",
            community_id="root",
            summary="summary root",
            full_content="full content root forbidden",
            rank=6.0,
        ),
    ]


def _make_entities() -> list[Entity]:
    return [
        Entity(id="e1", short_id="1", title="Entity1", community_ids=["l1", "p1"]),
        Entity(id="e2", short_id="2", title="Entity2", community_ids=["l2"]),
    ]


def _build_context_builder() -> LocalSearchMixedContext:
    entities = _make_entities()
    return LocalSearchMixedContext(
        entities=entities,
        communities=_make_communities(),
        community_reports=_make_reports(),
        entity_text_embeddings=object(),  # type: ignore[arg-type]
        text_embedder=object(),  # type: ignore[arg-type]
        text_units=[],
        relationships=[],
        covariates={
            "claims": [
                Covariate(
                    id="cov1",
                    short_id="c1",
                    subject_id="Entity1",
                    attributes={"description": "covariate A"},
                )
            ]
        },
        tokenizer=DummyTokenizer(),  # type: ignore[arg-type]
    )


def _selection_inputs():
    reports = _make_reports()
    communities = {c.short_id: c for c in _make_communities() if c.short_id}
    report_map = {r.community_id: r for r in reports}
    ranked = reports[:]
    return ranked, reports, communities, report_map


def test_leaf_detection_in_policy_inputs():
    ranked_matched, ranked_all, communities, report_map = _selection_inputs()
    result = select_community_reports(
        policy="leaf_only",
        ranked_matched_reports=ranked_matched,
        ranked_all_reports=ranked_all,
        communities_by_short_id=communities,
        report_by_community_id=report_map,
        max_tokens=10_000,
        token_counter=lambda _: 10,
        preserve_mode="fallback",
    )
    assert "l1" in [report.community_id for report in result.selected_reports]
    assert "l2" in [report.community_id for report in result.selected_reports]


def test_policy_leaf_only_selection():
    ranked_matched, ranked_all, communities, report_map = _selection_inputs()
    result = select_community_reports(
        policy="leaf_only",
        ranked_matched_reports=ranked_matched,
        ranked_all_reports=ranked_all,
        communities_by_short_id=communities,
        report_by_community_id=report_map,
        max_tokens=25,
        token_counter=lambda _: 10,
        preserve_mode="fallback",
    )
    selected_ids = [report.community_id for report in result.selected_reports]
    assert selected_ids[:2] == ["l1", "l2"]


def test_policy_leaf_then_parent_mix_replacement_and_warning():
    ranked_matched, ranked_all, communities, report_map = _selection_inputs()
    result = select_community_reports(
        policy="leaf_then_parent_mix",
        ranked_matched_reports=ranked_matched,
        ranked_all_reports=ranked_all,
        communities_by_short_id=communities,
        report_by_community_id=report_map,
        max_tokens=10,
        token_counter=lambda _: 10,
        preserve_mode="fallback",
    )
    selected_ids = [report.community_id for report in result.selected_reports]
    assert selected_ids == ["p1"]
    assert any("only one community" in warning for warning in result.warnings)


def test_policy_pyramid_selection_includes_leaf_first():
    ranked_matched, ranked_all, communities, report_map = _selection_inputs()
    result = select_community_reports(
        policy="pyramid",
        ranked_matched_reports=ranked_matched,
        ranked_all_reports=ranked_all,
        communities_by_short_id=communities,
        report_by_community_id=report_map,
        max_tokens=30,
        token_counter=lambda _: 10,
        preserve_mode="fallback",
    )
    selected_ids = [report.community_id for report in result.selected_reports]
    assert selected_ids[0] == "l1"
    assert "p1" in selected_ids


def test_policy_flat_ranked_selection():
    ranked_matched, ranked_all, communities, report_map = _selection_inputs()
    result = select_community_reports(
        policy="flat_ranked",
        ranked_matched_reports=ranked_matched,
        ranked_all_reports=ranked_all,
        communities_by_short_id=communities,
        report_by_community_id=report_map,
        max_tokens=20,
        token_counter=lambda _: 10,
        preserve_mode="fallback",
    )
    selected_ids = [report.community_id for report in result.selected_reports]
    assert selected_ids == ["l1", "l2"]


def test_history_on_off_and_overflow(monkeypatch):
    context_builder = _build_context_builder()
    entities = _make_entities()
    monkeypatch.setattr(
        "graphrag.query.structured_search.local_search.mixed_context.map_query_to_entities",
        lambda **_: entities,
    )
    history = ConversationHistory.from_list([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ])

    result_off = context_builder.build_context(
        query="hello",
        conversation_history=history,
        experimental_context_mode=True,
        experimental_community_policy="flat_ranked",
        experimental_history_enabled=False,
        experimental_covariate_enabled=False,
        max_context_tokens=200,
    )
    assert "Conversation History" not in result_off.context_chunks

    result_overflow = context_builder.build_context(
        query="hello",
        conversation_history=history,
        experimental_context_mode=True,
        experimental_community_policy="flat_ranked",
        experimental_history_enabled=True,
        experimental_covariate_enabled=False,
        max_context_tokens=5,
    )
    payload = result_overflow.context_records["experimental_context"].iloc[0].to_dict()
    assert payload["history_tokens"] == 0
    assert any("excluded" in warning for warning in payload["warnings"])


def test_covariate_on_off(monkeypatch):
    context_builder = _build_context_builder()
    entities = _make_entities()
    monkeypatch.setattr(
        "graphrag.query.structured_search.local_search.mixed_context.map_query_to_entities",
        lambda **_: entities,
    )
    result_on = context_builder.build_context(
        query="hello",
        experimental_context_mode=True,
        experimental_community_policy="flat_ranked",
        experimental_history_enabled=False,
        experimental_covariate_enabled=True,
        max_context_tokens=200,
    )
    payload_on = result_on.context_records["experimental_context"].iloc[0].to_dict()
    assert payload_on["covariate_tokens"] > 0

    result_off = context_builder.build_context(
        query="hello",
        experimental_context_mode=True,
        experimental_community_policy="flat_ranked",
        experimental_history_enabled=False,
        experimental_covariate_enabled=False,
        max_context_tokens=200,
    )
    payload_off = result_off.context_records["experimental_context"].iloc[0].to_dict()
    assert payload_off["covariate_tokens"] == 0


def test_experimental_payload_includes_trace_fields(monkeypatch):
    context_builder = _build_context_builder()
    entities = _make_entities()
    monkeypatch.setattr(
        "graphrag.query.structured_search.local_search.mixed_context.map_query_to_entities",
        lambda **_: entities,
    )
    result = context_builder.build_context(
        query="hello",
        experimental_context_mode=True,
        experimental_community_policy="flat_ranked",
        experimental_history_enabled=True,
        experimental_covariate_enabled=True,
        max_context_tokens=200,
        top_k_mapped_entities=7,
    )
    payload = result.context_records["experimental_context"].iloc[0].to_dict()
    assert payload["mapped_entities_top_k"] == 7
    assert payload["mapped_entities_count"] == 2
    assert payload["mapped_entity_titles"] == ["Entity1", "Entity2"]
    assert payload["community_summary_used"] is True
    assert isinstance(payload["selected_community_levels"], dict)
    assert "2" in payload["selected_community_levels"]


def test_assembled_context_excludes_full_content_and_entity_relationship_text_unit(monkeypatch):
    context_builder = _build_context_builder()
    entities = _make_entities()
    monkeypatch.setattr(
        "graphrag.query.structured_search.local_search.mixed_context.map_query_to_entities",
        lambda **_: entities,
    )
    result = context_builder.build_context(
        query="hello",
        experimental_context_mode=True,
        experimental_community_policy="flat_ranked",
        experimental_history_enabled=False,
        experimental_covariate_enabled=True,
        max_context_tokens=200,
    )
    context_text = str(result.context_chunks)
    assert "full content" not in context_text
    assert "-----Entities-----" not in context_text
    assert "-----Relationships-----" not in context_text
    assert "-----Sources" not in context_text


def test_community_summary_context_uses_summary_only(monkeypatch):
    context_builder = _build_context_builder()
    entities = _make_entities()
    monkeypatch.setattr(
        "graphrag.query.structured_search.local_search.mixed_context.map_query_to_entities",
        lambda **_: entities,
    )
    result = context_builder.build_context(
        query="hello",
        experimental_context_mode=True,
        experimental_community_policy="flat_ranked",
        experimental_history_enabled=False,
        experimental_covariate_enabled=False,
        max_context_tokens=200,
    )
    context_text = str(result.context_chunks)
    assert "summary l1" in context_text
    assert "full content l1 forbidden" not in context_text


def test_assembled_context_token_budget(monkeypatch):
    context_builder = _build_context_builder()
    entities = _make_entities()
    monkeypatch.setattr(
        "graphrag.query.structured_search.local_search.mixed_context.map_query_to_entities",
        lambda **_: entities,
    )
    result = context_builder.build_context(
        query="hello",
        experimental_context_mode=True,
        experimental_community_policy="flat_ranked",
        experimental_history_enabled=False,
        experimental_covariate_enabled=True,
        max_context_tokens=20,
    )
    payload = result.context_records["experimental_context"].iloc[0].to_dict()
    assert payload["assembled_context_tokens"] <= 20


def test_logging_payload_contains_required_fields(monkeypatch):
    context_builder = _build_context_builder()
    entities = _make_entities()
    monkeypatch.setattr(
        "graphrag.query.structured_search.local_search.mixed_context.map_query_to_entities",
        lambda **_: entities,
    )
    result = context_builder.build_context(
        query="hello",
        experimental_context_mode=True,
        experimental_community_policy="leaf_only",
        experimental_history_enabled=False,
        experimental_covariate_enabled=False,
        experimental_condition_id="cond_1",
        max_context_tokens=200,
    )
    payload = result.context_records["experimental_context"].iloc[0].to_dict()
    required_fields = {
        "condition_id",
        "community_policy",
        "policy_preserve_mode",
        "history_enabled",
        "covariate_enabled",
        "query",
        "selected_community_ids",
        "selected_community_ids_pre_filter",
        "dropped_community_ids",
        "primary_selected_ids",
        "fallback_selected_ids",
        "assembled_context",
        "community_tokens",
        "covariate_tokens",
        "history_tokens",
        "assembled_context_tokens",
        "warnings",
    }
    assert required_fields.issubset(set(payload.keys()))
    assert payload["condition_id"] == "cond_1"
    assert "community_context" not in payload
    assert "history_context" not in payload
    assert "covariate_context" not in payload


def test_policy_preserve_mode_strict_disables_fallback(monkeypatch):
    context_builder = _build_context_builder()
    entities = _make_entities()
    monkeypatch.setattr(
        "graphrag.query.structured_search.local_search.mixed_context.map_query_to_entities",
        lambda **_: entities,
    )
    result = context_builder.build_context(
        query="hello",
        experimental_context_mode=True,
        experimental_community_policy="leaf_only",
        experimental_policy_preserve_mode="strict",
        experimental_history_enabled=False,
        experimental_covariate_enabled=False,
        max_context_tokens=200,
    )
    payload = result.context_records["experimental_context"].iloc[0].to_dict()
    assert payload["fallback_selected_ids"] == []
    assert payload["primary_selected_ids"] == ["l1", "l2"]
    assert any("strict" in warning for warning in payload["warnings"])


def test_payload_selected_ids_track_inserted_reports(monkeypatch):
    context_builder = _build_context_builder()
    entities = _make_entities()
    monkeypatch.setattr(
        "graphrag.query.structured_search.local_search.mixed_context.map_query_to_entities",
        lambda **_: entities,
    )
    result = context_builder.build_context(
        query="hello",
        experimental_context_mode=True,
        experimental_community_policy="flat_ranked",
        experimental_history_enabled=False,
        experimental_covariate_enabled=False,
        min_community_rank=8,
        max_context_tokens=200,
    )
    payload = result.context_records["experimental_context"].iloc[0].to_dict()
    assert payload["selected_community_ids"] == ["l1", "l2"]
    assert payload["dropped_community_ids"] == ["p1", "root"]


def test_experimental_payload_keeps_only_assembled_context(monkeypatch):
    context_builder = _build_context_builder()
    entities = _make_entities()
    monkeypatch.setattr(
        "graphrag.query.structured_search.local_search.mixed_context.map_query_to_entities",
        lambda **_: entities,
    )
    history = ConversationHistory.from_list(
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
    )
    result = context_builder.build_context(
        query="hello",
        conversation_history=history,
        experimental_context_mode=True,
        experimental_community_policy="flat_ranked",
        experimental_history_enabled=True,
        experimental_covariate_enabled=True,
        max_context_tokens=200,
    )
    payload = result.context_records["experimental_context"].iloc[0].to_dict()
    assert "assembled_context" in payload
    assert "community_context" not in payload
    assert "history_context" not in payload
    assert "covariate_context" not in payload


def test_experimental_history_toggle_does_not_affect_entity_retrieval_or_selected_communities(
    monkeypatch,
):
    context_builder = _build_context_builder()
    recorded_queries: list[str] = []
    entities = _make_entities()

    def _capture_map_query_to_entities(**kwargs):
        recorded_queries.append(kwargs["query"])
        return entities

    monkeypatch.setattr(
        "graphrag.query.structured_search.local_search.mixed_context.map_query_to_entities",
        _capture_map_query_to_entities,
    )

    history = ConversationHistory.from_list(
        [
            {"role": "user", "content": "older-user-q"},
            {"role": "assistant", "content": "older-assistant-a"},
        ]
    )

    result_history_off = context_builder.build_context(
        query="hello",
        conversation_history=history,
        experimental_context_mode=True,
        experimental_community_policy="flat_ranked",
        experimental_history_enabled=False,
        experimental_covariate_enabled=False,
        max_context_tokens=200,
    )
    result_history_on = context_builder.build_context(
        query="hello",
        conversation_history=history,
        experimental_context_mode=True,
        experimental_community_policy="flat_ranked",
        experimental_history_enabled=True,
        experimental_covariate_enabled=False,
        max_context_tokens=200,
    )

    payload_off = result_history_off.context_records["experimental_context"].iloc[0].to_dict()
    payload_on = result_history_on.context_records["experimental_context"].iloc[0].to_dict()
    assert payload_off["selected_community_ids"] == payload_on["selected_community_ids"]
    assert recorded_queries == ["hello", "hello"]
