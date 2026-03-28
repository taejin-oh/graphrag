# Copyright (C) 2026 Microsoft Corporation.
# Licensed under the MIT License

from __future__ import annotations

import pytest

from graphrag.data_model.community import Community
from graphrag.data_model.community_report import CommunityReport
from graphrag.data_model.covariate import Covariate
from graphrag.data_model.entity import Entity
from graphrag.query.context_builder.conversation_history import ConversationHistory
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


@pytest.mark.parametrize(
    ("policy", "covariate_enabled"),
    [
        ("flat_ranked", False),
        ("flat_ranked", True),
        ("leaf_only", False),
        ("leaf_then_parent_mix", False),
        ("pyramid", False),
    ],
)
def test_experimental_integration_history_toggle_keeps_retrieval_and_selection_stable(
    monkeypatch, policy: str, covariate_enabled: bool
):
    context_builder = _build_context_builder()
    entities = _make_entities()
    recorded_queries: list[str] = []

    def _capture_map_query_to_entities(**kwargs):
        recorded_queries.append(kwargs["query"])
        return entities

    monkeypatch.setattr(
        "graphrag.query.structured_search.local_search.mixed_context.map_query_to_entities",
        _capture_map_query_to_entities,
    )

    history = ConversationHistory.from_list(
        [
            {"role": "user", "content": "old q1"},
            {"role": "assistant", "content": "old a1"},
            {"role": "user", "content": "old q2"},
        ]
    )
    base_args = dict(
        query="integration hello",
        conversation_history=history,
        experimental_context_mode=True,
        experimental_community_policy=policy,
        experimental_covariate_enabled=covariate_enabled,
        max_context_tokens=200,
    )
    result_history_off = context_builder.build_context(
        **base_args,
        experimental_history_enabled=False,
    )
    result_history_on = context_builder.build_context(
        **base_args,
        experimental_history_enabled=True,
    )

    payload_off = result_history_off.context_records["experimental_context"].iloc[0].to_dict()
    payload_on = result_history_on.context_records["experimental_context"].iloc[0].to_dict()
    assert payload_off["selected_community_ids"] == payload_on["selected_community_ids"]
    assert recorded_queries == ["integration hello", "integration hello"]


def test_experimental_integration_payload_keeps_only_assembled_context(monkeypatch):
    context_builder = _build_context_builder()
    entities = _make_entities()
    monkeypatch.setattr(
        "graphrag.query.structured_search.local_search.mixed_context.map_query_to_entities",
        lambda **_: entities,
    )

    result = context_builder.build_context(
        query="integration hello",
        conversation_history=ConversationHistory.from_list(
            [{"role": "user", "content": "history"}]
        ),
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
