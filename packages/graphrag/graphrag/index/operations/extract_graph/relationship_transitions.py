# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Build temporal target-transition events from merged relationship edges."""

from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from graphrag.data_model.schemas import (
    CHUNK_INDEX_IN_CONVERSATION,
    CHUNK_INDEX_IN_DOCUMENT,
    CONVERSATION_ID,
    END_TURN_INDEX,
    START_TURN_INDEX,
    TURN_TIMESTAMP_END,
    TURN_TIMESTAMP_START,
)


def build_relationship_transitions(
    relationships: pd.DataFrame,
    text_units: pd.DataFrame,
    text_unit_id_column: str = "id",
) -> pd.DataFrame:
    """Build transition events where a source's target changes over time.

    The function assumes ``relationships`` is already merged and contains
    ``source``, ``target``, ``description``, and ``text_unit_ids`` fields.

    Returns a dataframe with one row per detected target change event.
    """
    if relationships.empty or text_units.empty or text_unit_id_column not in text_units:
        return _empty_transitions_df()

    temporal_index = _build_temporal_index(text_units, text_unit_id_column)
    events: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    source_slot_registry: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    ordered_rows = sorted(
        relationships.itertuples(index=False),
        key=lambda row: _row_first_seen_sort_key(row, temporal_index),
    )

    for row in ordered_rows:
        source = str(getattr(row, "source", "") or "")
        target = str(getattr(row, "target", "") or "")
        text_unit_ids = _as_list(getattr(row, "text_unit_ids", []))
        if not source or not target or not text_unit_ids:
            continue

        relation_tokens = _relation_tokens(
            source=source,
            target=target,
            description=getattr(row, "description", None),
        )
        relation_slot = _assign_relation_slot(
            source=source,
            relation_tokens=relation_tokens,
            registry=source_slot_registry,
        )
        sorted_ids = sorted(
            [str(text_unit_id) for text_unit_id in text_unit_ids],
            key=lambda tid: _temporal_sort_key(tid, temporal_index),
        )

        for text_unit_id in sorted_ids:
            metadata = temporal_index.get(text_unit_id, {})
            events[(source, relation_slot)].append({
                "source": source,
                "target": target,
                "relation_slot": relation_slot,
                "text_unit_id": text_unit_id,
                "conversation_id": metadata.get(CONVERSATION_ID),
                "turn_index": metadata.get(START_TURN_INDEX),
                "turn_index_end": metadata.get(END_TURN_INDEX),
                "timestamp": metadata.get(TURN_TIMESTAMP_START),
                "timestamp_end": metadata.get(TURN_TIMESTAMP_END),
            })

    transitions: list[dict[str, object]] = []
    for (source, relation_slot), source_events in events.items():
        ordered = sorted(
            source_events,
            key=lambda item: _temporal_sort_key(str(item["text_unit_id"]), temporal_index),
        )
        prev_target: str | None = None
        prev_event: dict[str, object] | None = None
        change_index = 0
        for event in ordered:
            current_target = str(event["target"])
            if prev_target is not None and current_target != prev_target and prev_event:
                transitions.append({
                    "source": source,
                    "relation_slot": relation_slot,
                    "from_target": prev_target,
                    "to_target": current_target,
                    "changed_at_text_unit_id": event["text_unit_id"],
                    "changed_at_turn_index": event["turn_index"],
                    "changed_at_timestamp": event["timestamp"],
                    "previous_text_unit_id": prev_event["text_unit_id"],
                    "previous_turn_index": prev_event["turn_index"],
                    "previous_timestamp": prev_event["timestamp"],
                    "change_index": change_index,
                })
                change_index += 1
            prev_target = current_target
            prev_event = event

    if not transitions:
        return _empty_transitions_df()

    result = pd.DataFrame(transitions)
    result.sort_values(
        by=["source", "relation_slot", "changed_at_turn_index", "changed_at_timestamp"],
        inplace=True,
        na_position="last",
    )
    result.reset_index(drop=True, inplace=True)
    return result


def _build_temporal_index(
    text_units: pd.DataFrame,
    text_unit_id_column: str,
) -> dict[str, dict[str, str | int | None]]:
    temporal_index: dict[str, dict[str, str | int | None]] = {}
    for _, row in text_units.iterrows():
        text_unit_id = _safe_str(row.get(text_unit_id_column))
        if not text_unit_id:
            continue
        temporal_index[text_unit_id] = {
            CONVERSATION_ID: _safe_str(row.get(CONVERSATION_ID)),
            START_TURN_INDEX: _safe_int(row.get(START_TURN_INDEX)),
            END_TURN_INDEX: _safe_int(row.get(END_TURN_INDEX)),
            TURN_TIMESTAMP_START: _safe_str(row.get(TURN_TIMESTAMP_START)),
            TURN_TIMESTAMP_END: _safe_str(row.get(TURN_TIMESTAMP_END)),
            CHUNK_INDEX_IN_CONVERSATION: _safe_int(row.get(CHUNK_INDEX_IN_CONVERSATION)),
            CHUNK_INDEX_IN_DOCUMENT: _safe_int(row.get(CHUNK_INDEX_IN_DOCUMENT)),
        }
    return temporal_index


def _assign_relation_slot(
    source: str,
    relation_tokens: set[str],
    registry: dict[str, list[tuple[str, set[str]]]],
    threshold: float = 0.3,
) -> str:
    existing_slots = registry[source]
    for slot_name, slot_tokens in existing_slots:
        if _jaccard_similarity(relation_tokens, slot_tokens) >= threshold:
            return slot_name
    slot_name = f"slot_{len(existing_slots)}"
    existing_slots.append((slot_name, relation_tokens))
    return slot_name


def _relation_tokens(description: object, source: str, target: str) -> set[str]:
    normalized = _normalize_description(description, source=source, target=target)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return {
        _canonical_relation_token(token)
        for token in tokens
        if len(token) > 2 and token not in _STOPWORDS
    }


def _normalize_description(description: object, source: str, target: str) -> str:
    if isinstance(description, list):
        text = " ".join(str(item).strip() for item in description if str(item).strip())
    else:
        text = str(description or "").strip()
    if not text:
        return "unknown"
    lowered = text.lower()
    lowered = _remove_entity_phrase(lowered, source.lower())
    lowered = _remove_entity_phrase(lowered, target.lower())
    return " ".join(lowered.split())[:120]


def _remove_entity_phrase(text: str, phrase: str) -> str:
    cleaned = phrase.strip()
    if not cleaned:
        return text
    escaped = r"\s+".join(re.escape(part) for part in cleaned.split() if part)
    if not escaped:
        return text
    pattern = re.compile(rf"\b{escaped}\b")
    return pattern.sub(" ", text)


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _canonical_relation_token(token: str) -> str:
    if token in {"join", "joined", "joins", "work", "works", "worked", "working"}:
        return "employment"
    if token in {"employ", "employed", "employee", "employment"}:
        return "employment"
    if token in {"live", "lives", "living", "reside", "resides", "resident"}:
        return "residence"
    return token


def _row_first_seen_sort_key(
    row: object,
    temporal_index: dict[str, dict[str, str | int | None]],
) -> tuple[str, int, str, int, int, str]:
    text_unit_ids = _as_list(getattr(row, "text_unit_ids", []))
    if not text_unit_ids:
        return ("~", 10**12, "~", 10**12, 10**12, "~")
    first_id = min(
        (str(text_unit_id) for text_unit_id in text_unit_ids),
        key=lambda tid: _temporal_sort_key(tid, temporal_index),
    )
    return _temporal_sort_key(first_id, temporal_index)


def _as_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _temporal_sort_key(
    text_unit_id: str,
    temporal_index: dict[str, dict[str, str | int | None]],
) -> tuple[str, int, str, int, int, str]:
    metadata = temporal_index.get(text_unit_id, {})
    return (
        str(metadata.get(CONVERSATION_ID) or "~"),
        _sortable_int(metadata.get(START_TURN_INDEX)),
        str(metadata.get(TURN_TIMESTAMP_START) or "~"),
        _sortable_int(metadata.get(CHUNK_INDEX_IN_CONVERSATION)),
        _sortable_int(metadata.get(CHUNK_INDEX_IN_DOCUMENT)),
        text_unit_id,
    )


def _sortable_int(value: object) -> int:
    parsed = _safe_int(value)
    return parsed if parsed is not None else 10**12


def _safe_int(value: object) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _empty_transitions_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source",
            "relation_slot",
            "from_target",
            "to_target",
            "changed_at_text_unit_id",
            "changed_at_turn_index",
            "changed_at_timestamp",
            "previous_text_unit_id",
            "previous_turn_index",
            "previous_timestamp",
            "change_index",
        ]
    )


_STOPWORDS = {
    "and",
    "are",
    "because",
    "for",
    "from",
    "has",
    "have",
    "into",
    "now",
    "that",
    "the",
    "their",
    "then",
    "this",
    "was",
    "were",
    "with",
}
