# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

import pandas as pd

from graphrag.index.operations.extract_graph.relationship_transitions import (
    build_relationship_transitions,
)


def test_build_relationship_transitions_detects_target_change():
    relationships = pd.DataFrame(
        [
            {
                "source": "ALICE",
                "target": "COMPANY_A",
                "description": ["ALICE works at COMPANY_A as engineer"],
                "text_unit_ids": ["t1", "t2"],
            },
            {
                "source": "ALICE",
                "target": "COMPANY_B",
                "description": ["ALICE works at COMPANY_B as engineer"],
                "text_unit_ids": ["t3"],
            },
        ]
    )

    text_units = pd.DataFrame(
        [
            {
                "id": "t1",
                "conversation_id": "c1",
                "start_turn_index": 1,
                "end_turn_index": 1,
                "turn_timestamp_start": "2026-01-01T00:00:00Z",
                "turn_timestamp_end": "2026-01-01T00:00:00Z",
                "chunk_index_in_conversation": 0,
                "chunk_index_in_document": 0,
            },
            {
                "id": "t2",
                "conversation_id": "c1",
                "start_turn_index": 2,
                "end_turn_index": 2,
                "turn_timestamp_start": "2026-01-01T00:01:00Z",
                "turn_timestamp_end": "2026-01-01T00:01:00Z",
                "chunk_index_in_conversation": 1,
                "chunk_index_in_document": 0,
            },
            {
                "id": "t3",
                "conversation_id": "c1",
                "start_turn_index": 3,
                "end_turn_index": 3,
                "turn_timestamp_start": "2026-01-01T00:02:00Z",
                "turn_timestamp_end": "2026-01-01T00:02:00Z",
                "chunk_index_in_conversation": 2,
                "chunk_index_in_document": 0,
            },
        ]
    )

    transitions = build_relationship_transitions(relationships, text_units)

    assert len(transitions) == 1
    row = transitions.iloc[0]
    assert row["source"] == "ALICE"
    assert row["from_target"] == "COMPANY_A"
    assert row["to_target"] == "COMPANY_B"
    assert row["changed_at_turn_index"] == 3


def test_build_relationship_transitions_handles_wording_drift():
    relationships = pd.DataFrame(
        [
            {
                "source": "ALICE",
                "target": "COMPANY_A",
                "description": ["ALICE works at COMPANY_A as engineer"],
                "text_unit_ids": ["t1"],
            },
            {
                "source": "ALICE",
                "target": "COMPANY_B",
                "description": ["ALICE now works for COMPANY_B in engineering"],
                "text_unit_ids": ["t2"],
            },
        ]
    )
    text_units = pd.DataFrame(
        [
            {
                "id": "t1",
                "conversation_id": "c1",
                "start_turn_index": 1,
                "end_turn_index": 1,
                "turn_timestamp_start": "2026-01-01T00:00:00Z",
                "turn_timestamp_end": "2026-01-01T00:00:00Z",
                "chunk_index_in_conversation": 0,
                "chunk_index_in_document": 0,
            },
            {
                "id": "t2",
                "conversation_id": "c1",
                "start_turn_index": 2,
                "end_turn_index": 2,
                "turn_timestamp_start": "2026-01-01T00:01:00Z",
                "turn_timestamp_end": "2026-01-01T00:01:00Z",
                "chunk_index_in_conversation": 1,
                "chunk_index_in_document": 0,
            },
        ]
    )

    transitions = build_relationship_transitions(relationships, text_units)

    assert len(transitions) == 1
    row = transitions.iloc[0]
    assert row["from_target"] == "COMPANY_A"
    assert row["to_target"] == "COMPANY_B"


def test_build_relationship_transitions_handles_joined_to_works():
    relationships = pd.DataFrame(
        [
            {
                "source": "ALICE",
                "target": "COMPANY_A",
                "description": ["ALICE joined COMPANY_A in 2020"],
                "text_unit_ids": ["t1"],
            },
            {
                "source": "ALICE",
                "target": "COMPANY_B",
                "description": ["ALICE now works for COMPANY_B"],
                "text_unit_ids": ["t2"],
            },
        ]
    )
    text_units = pd.DataFrame(
        [
            {
                "id": "t1",
                "conversation_id": "c1",
                "start_turn_index": 1,
                "end_turn_index": 1,
                "turn_timestamp_start": "2026-01-01T00:00:00Z",
                "turn_timestamp_end": "2026-01-01T00:00:00Z",
                "chunk_index_in_conversation": 0,
                "chunk_index_in_document": 0,
            },
            {
                "id": "t2",
                "conversation_id": "c1",
                "start_turn_index": 2,
                "end_turn_index": 2,
                "turn_timestamp_start": "2026-01-01T00:01:00Z",
                "turn_timestamp_end": "2026-01-01T00:01:00Z",
                "chunk_index_in_conversation": 1,
                "chunk_index_in_document": 0,
            },
        ]
    )

    transitions = build_relationship_transitions(relationships, text_units)
    assert len(transitions) == 1


def test_build_relationship_transitions_keeps_distinct_relation_slots():
    relationships = pd.DataFrame(
        [
            {
                "source": "ALICE",
                "target": "COMPANY_A",
                "description": ["ALICE works at COMPANY_A as engineer"],
                "text_unit_ids": ["t1"],
            },
            {
                "source": "ALICE",
                "target": "SEOUL",
                "description": ["ALICE lives in SEOUL"],
                "text_unit_ids": ["t2"],
            },
        ]
    )
    text_units = pd.DataFrame(
        [
            {
                "id": "t1",
                "conversation_id": "c1",
                "start_turn_index": 1,
                "end_turn_index": 1,
                "turn_timestamp_start": "2026-01-01T00:00:00Z",
                "turn_timestamp_end": "2026-01-01T00:00:00Z",
                "chunk_index_in_conversation": 0,
                "chunk_index_in_document": 0,
            },
            {
                "id": "t2",
                "conversation_id": "c1",
                "start_turn_index": 2,
                "end_turn_index": 2,
                "turn_timestamp_start": "2026-01-01T00:01:00Z",
                "turn_timestamp_end": "2026-01-01T00:01:00Z",
                "chunk_index_in_conversation": 1,
                "chunk_index_in_document": 0,
            },
        ]
    )

    transitions = build_relationship_transitions(relationships, text_units)
    assert transitions.empty


def test_build_relationship_transitions_drops_events_without_temporal_metadata():
    relationships = pd.DataFrame(
        [
            {
                "source": "ALICE",
                "target": "COMPANY_A",
                "description": ["ALICE works at COMPANY_A"],
                "text_unit_ids": ["missing_t1"],
            },
            {
                "source": "ALICE",
                "target": "COMPANY_B",
                "description": ["ALICE works at COMPANY_B"],
                "text_unit_ids": ["missing_t2"],
            },
        ]
    )
    text_units = pd.DataFrame(
        [
            {
                "id": "known_t1",
                "conversation_id": "c1",
                "start_turn_index": 1,
                "end_turn_index": 1,
                "turn_timestamp_start": "2026-01-01T00:00:00Z",
                "turn_timestamp_end": "2026-01-01T00:00:00Z",
                "chunk_index_in_conversation": 0,
                "chunk_index_in_document": 0,
            }
        ]
    )

    transitions = build_relationship_transitions(relationships, text_units)
    assert transitions.empty
