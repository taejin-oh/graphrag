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


def test_sort_context_prioritizes_temporal_order_over_degree():
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
    assert "start_turn_index" in header_line
    assert "end_turn_index" in header_line
    assert "turn_timestamp_start" in header_line
    assert "turn_timestamp_end" in header_line
