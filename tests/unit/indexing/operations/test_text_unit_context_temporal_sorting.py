# Copyright (C) 2026 Microsoft Corporation.
# Licensed under the MIT License

import pandas as pd

import graphrag.data_model.schemas as schemas
from graphrag.index.operations.summarize_communities.text_unit_context.prep_text_units import (
    prep_text_units,
)
from graphrag.index.operations.summarize_communities.text_unit_context.sort_context import (
    sort_context,
)
from graphrag.index.operations.summarize_communities.text_unit_context.context_builder import (
    build_local_context,
    build_level_context,
)


class _FakeTokenizer:
    def num_tokens(self, text: str) -> int:
        return len(text)


def test_prep_text_units_includes_temporal_fields_in_all_details():
    text_units = pd.DataFrame(
        [
            {
                "id": "t1",
                "human_readable_id": 1,
                "text": "first",
                "start_turn_index": 1,
                "end_turn_index": 1,
                "turn_timestamp_start": "2026-01-01T09:00:00Z",
                "turn_timestamp_end": "2026-01-01T09:00:30Z",
                "chunk_index_in_conversation": 0,
            }
        ]
    )
    nodes = pd.DataFrame(
        [
            {
                "id": "n1",
                "title": "Entity",
                "community": 1,
                "degree": 3,
                "text_unit_ids": ["t1"],
            }
        ]
    )

    out = prep_text_units(text_units, nodes)
    details = out.iloc[0][schemas.ALL_DETAILS]

    assert details[schemas.START_TURN_INDEX] == 1
    assert details[schemas.END_TURN_INDEX] == 1
    assert details[schemas.TURN_TIMESTAMP_START] == "2026-01-01T09:00:00Z"
    assert details[schemas.TURN_TIMESTAMP_END] == "2026-01-01T09:00:30Z"


def test_sort_context_prioritizes_entity_degree_over_temporal_order():
    context = [
        {
            "id": 2,
            "text": "later but high degree",
            "entity_degree": 100,
            "start_turn_index": 5,
            "turn_timestamp_start": "2026-01-01T10:00:00Z",
            "chunk_index_in_conversation": 1,
        },
        {
            "id": 1,
            "text": "earlier lower degree",
            "entity_degree": 1,
            "start_turn_index": 1,
            "turn_timestamp_start": "2026-01-01T09:00:00Z",
            "chunk_index_in_conversation": 0,
        },
    ]

    out = sort_context(context, tokenizer=_FakeTokenizer())

    first_data_line = out.splitlines()[2]
    assert first_data_line.startswith("2,")


def test_sort_context_uses_temporal_order_as_tie_breaker():
    context = [
        {
            "id": 2,
            "text": "later",
            "entity_degree": 10,
            "start_turn_index": 5,
            "turn_timestamp_start": "2026-01-01T10:00:00Z",
            "chunk_index_in_conversation": 1,
        },
        {
            "id": 1,
            "text": "earlier",
            "entity_degree": 10,
            "start_turn_index": 1,
            "turn_timestamp_start": "2026-01-01T09:00:00Z",
            "chunk_index_in_conversation": 0,
        },
    ]

    out = sort_context(context, tokenizer=_FakeTokenizer())

    first_data_line = out.splitlines()[2]
    assert first_data_line.startswith("1,")


def test_build_local_context_emits_temporal_columns_into_sources_context():
    community_membership = pd.DataFrame(
        [
            {
                "community": 1,
                "level": 1,
                "text_unit_ids": ["t1", "t2"],
            }
        ]
    )
    text_units = pd.DataFrame(
        [
            {
                "id": "t1",
                "human_readable_id": 1,
                "text": "first",
                "start_turn_index": 1,
                "end_turn_index": 1,
                "turn_timestamp_start": "2026-01-01T09:00:00Z",
                "turn_timestamp_end": "2026-01-01T09:00:30Z",
                "chunk_index_in_conversation": 0,
            },
            {
                "id": "t2",
                "human_readable_id": 2,
                "text": "second",
                "start_turn_index": 2,
                "end_turn_index": 2,
                "turn_timestamp_start": "2026-01-01T10:00:00Z",
                "turn_timestamp_end": "2026-01-01T10:00:30Z",
                "chunk_index_in_conversation": 1,
            },
        ]
    )
    nodes = pd.DataFrame(
        [
            {
                "id": "n1",
                "title": "Entity",
                "community": 1,
                "degree": 3,
                "text_unit_ids": ["t1", "t2"],
            }
        ]
    )

    out = build_local_context(
        community_membership_df=community_membership,
        text_units_df=text_units,
        node_df=nodes,
        tokenizer=_FakeTokenizer(),
    )

    context_string = out.iloc[0][schemas.CONTEXT_STRING]
    header_line = context_string.splitlines()[1]
    first_data_line = context_string.splitlines()[2]
    assert "start_turn_index" in header_line
    assert "end_turn_index" in header_line
    assert "turn_timestamp_start" in header_line
    assert "turn_timestamp_end" in header_line
    assert "chunk_index_in_conversation" in header_line
    assert "2026-01-01T09:00:00Z" in first_data_line
    assert ",1,1," in first_data_line


def test_build_local_context_emits_relationship_transition_section():
    community_membership = pd.DataFrame(
        [
            {
                "community": 1,
                "level": 1,
                "entity_ids": ["n1"],
                "text_unit_ids": ["t1", "t2"],
            }
        ]
    )
    text_units = pd.DataFrame(
        [
            {
                "id": "t1",
                "human_readable_id": 1,
                "text": "alice worked at company a",
                "start_turn_index": 1,
                "end_turn_index": 1,
                "turn_timestamp_start": "2026-01-01T09:00:00Z",
                "turn_timestamp_end": "2026-01-01T09:00:30Z",
                "chunk_index_in_conversation": 0,
            },
            {
                "id": "t2",
                "human_readable_id": 2,
                "text": "alice now works at company b",
                "start_turn_index": 2,
                "end_turn_index": 2,
                "turn_timestamp_start": "2026-01-01T10:00:00Z",
                "turn_timestamp_end": "2026-01-01T10:00:30Z",
                "chunk_index_in_conversation": 1,
            },
        ]
    )
    nodes = pd.DataFrame(
        [
            {
                "id": "n1",
                "title": "ALICE",
                "community": 1,
                "degree": 3,
                "text_unit_ids": ["t1", "t2"],
            }
        ]
    )
    transitions = pd.DataFrame(
        [
            {
                "source": "ALICE",
                "relation_slot": "slot_0",
                "from_target": "COMPANY_A",
                "to_target": "COMPANY_B",
                "changed_at_text_unit_id": "t2",
                "changed_at_turn_index": 2,
                "changed_at_timestamp": "2026-01-01T10:00:00Z",
                "previous_text_unit_id": "t1",
                "previous_turn_index": 1,
                "previous_timestamp": "2026-01-01T09:00:00Z",
                "change_index": 0,
                "conversation_id": "c1",
            }
        ]
    )

    out = build_local_context(
        community_membership_df=community_membership,
        text_units_df=text_units,
        node_df=nodes,
        tokenizer=_FakeTokenizer(),
        relationship_transitions_df=transitions,
    )

    context_string = out.iloc[0][schemas.CONTEXT_STRING]
    assert "-----RELATIONSHIP_TRANSITIONS-----" in context_string
    assert "ALICE,slot_0,COMPANY_A,COMPANY_B" in context_string


def test_build_local_context_maps_entity_ids_to_titles_for_transition_fallback():
    community_membership = pd.DataFrame(
        [
            {
                "community": 1,
                "level": 1,
                "entity_ids": ["n1"],
                "text_unit_ids": ["t1"],
            }
        ]
    )
    text_units = pd.DataFrame(
        [
            {
                "id": "t1",
                "human_readable_id": 1,
                "text": "alice moved jobs",
                "start_turn_index": 1,
                "end_turn_index": 1,
                "turn_timestamp_start": "2026-01-01T09:00:00Z",
                "turn_timestamp_end": "2026-01-01T09:00:30Z",
                "chunk_index_in_conversation": 0,
            }
        ]
    )
    nodes = pd.DataFrame(
        [
            {
                "id": "n1",
                "title": "ALICE",
                "community": 1,
                "degree": 3,
                "text_unit_ids": ["t1"],
            }
        ]
    )
    transitions = pd.DataFrame(
        [
            {
                "source": "ALICE",
                "relation_slot": "slot_0",
                "from_target": "COMPANY_A",
                "to_target": "COMPANY_B",
                "changed_at_text_unit_id": "unknown_t2",
                "changed_at_turn_index": 2,
                "changed_at_timestamp": "2026-01-01T10:00:00Z",
                "previous_text_unit_id": "unknown_t1",
                "previous_turn_index": 1,
                "previous_timestamp": "2026-01-01T09:00:00Z",
                "change_index": 0,
                "conversation_id": "c1",
            }
        ]
    )

    out = build_local_context(
        community_membership_df=community_membership,
        text_units_df=text_units,
        node_df=nodes,
        tokenizer=_FakeTokenizer(),
        relationship_transitions_df=transitions,
    )

    context_string = out.iloc[0][schemas.CONTEXT_STRING]
    assert "-----RELATIONSHIP_TRANSITIONS-----" in context_string
    assert "ALICE,slot_0,COMPANY_A,COMPANY_B" in context_string


def test_sort_context_limits_transition_rows_under_token_budget():
    context = [
        {
            "id": 1,
            "text": "short source",
            "entity_degree": 1,
            "start_turn_index": 1,
            "turn_timestamp_start": "2026-01-01T09:00:00Z",
            "chunk_index_in_conversation": 0,
        }
    ]
    transitions = [
        {
            "source": "ALICE",
            "relation_slot": "slot_0",
            "from_target": "A",
            "to_target": "B",
            "changed_at_turn_index": 1,
            "changed_at_timestamp": "2026-01-01T09:00:00Z",
        },
        {
            "source": "ALICE",
            "relation_slot": "slot_0",
            "from_target": "B",
            "to_target": "C",
            "changed_at_turn_index": 2,
            "changed_at_timestamp": "2026-01-01T10:00:00Z",
        },
    ]

    out = sort_context(
        context,
        tokenizer=_FakeTokenizer(),
        transition_records=transitions,
        max_context_tokens=320,
    )

    assert "-----RELATIONSHIP_TRANSITIONS-----" in out
    assert "ALICE,slot_0,A,B" in out
    assert "ALICE,slot_0,B,C" not in out


def test_build_level_context_remaining_path_keeps_transition_records():
    local_context = pd.DataFrame(
        [
            {
                "community": 1,
                "level": 0,
                "all_context": [
                    {
                        "id": 1,
                        "text": "source text",
                        "entity_degree": 10,
                        "start_turn_index": 1,
                        "turn_timestamp_start": "2026-01-01T09:00:00Z",
                        "chunk_index_in_conversation": 0,
                    }
                ],
                "transition_records": [
                    {
                        "source": "ALICE",
                        "relation_slot": "slot_0",
                        "from_target": "COMPANY_A",
                        "to_target": "COMPANY_B",
                        "changed_at_turn_index": 2,
                        "changed_at_timestamp": "2026-01-01T10:00:00Z",
                    }
                ],
                "context_string": "x" * 999,
                "context_size": 999,
                "context_exceed_limit": True,
            },
            {
                "community": 2,
                "level": 1,
                "all_context": [],
                "transition_records": [],
                "context_string": "sub",
                "context_size": 3,
                "context_exceed_limit": False,
            },
        ]
    )
    hierarchy = pd.DataFrame(
        [{"community": 1, "level": 0, "sub_community": 999}]
    )
    report_df = pd.DataFrame(
        [{"community": 2, "level": 1, "full_content": "child report"}]
    )

    out = build_level_context(
        report_df=report_df,
        community_hierarchy_df=hierarchy,
        local_context_df=local_context,
        level=0,
        tokenizer=_FakeTokenizer(),
        max_context_tokens=1_000,
    )

    value = out.iloc[0][schemas.CONTEXT_STRING]
    assert "-----RELATIONSHIP_TRANSITIONS-----" in value
    assert "ALICE,slot_0,COMPANY_A,COMPANY_B" in value


def test_build_level_context_substitution_path_appends_transition_records():
    local_context = pd.DataFrame(
        [
            {
                "community": 1,
                "level": 0,
                "all_context": [
                    {
                        "id": 1,
                        "text": "parent source text",
                        "entity_degree": 5,
                        "start_turn_index": 1,
                        "turn_timestamp_start": "2026-01-01T09:00:00Z",
                        "chunk_index_in_conversation": 0,
                    }
                ],
                "transition_records": [
                    {
                        "source": "ALICE",
                        "relation_slot": "slot_0",
                        "from_target": "COMPANY_A",
                        "to_target": "COMPANY_B",
                        "changed_at_turn_index": 2,
                        "changed_at_timestamp": "2026-01-01T10:00:00Z",
                    }
                ],
                "context_string": "x" * 999,
                "context_size": 999,
                "context_exceed_limit": True,
            },
            {
                "community": 2,
                "level": 1,
                "all_context": [],
                "transition_records": [],
                "context_string": "child context",
                "context_size": 12,
                "context_exceed_limit": False,
            },
        ]
    )
    hierarchy = pd.DataFrame([{"community": 1, "level": 0, "sub_community": 2}])
    report_df = pd.DataFrame(
        [{"community": 2, "level": 1, "full_content": "child report"}]
    )

    out = build_level_context(
        report_df=report_df,
        community_hierarchy_df=hierarchy,
        local_context_df=local_context,
        level=0,
        tokenizer=_FakeTokenizer(),
        max_context_tokens=1_000,
    )

    value = out.iloc[0][schemas.CONTEXT_STRING]
    assert "child report" in value
    assert "-----RELATIONSHIP_TRANSITIONS-----" in value
