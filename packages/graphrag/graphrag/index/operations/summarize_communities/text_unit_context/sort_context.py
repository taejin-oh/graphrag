# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Sort local context by total degree of associated nodes in descending order."""

import logging

import pandas as pd
from graphrag_llm.tokenizer import Tokenizer

import graphrag.data_model.schemas as schemas

logger = logging.getLogger(__name__)


def _sortable_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 10**12


def _temporal_sort_key(unit: dict) -> tuple[int, str, int]:
    return (
        _sortable_int(unit.get(schemas.START_TURN_INDEX)),
        str(unit.get(schemas.TURN_TIMESTAMP_START) or "~"),
        _sortable_int(unit.get(schemas.CHUNK_INDEX_IN_CONVERSATION)),
    )


def get_context_string(
    text_units: list[dict],
    sub_community_reports: list[dict] | None = None,
    transition_records: list[dict] | None = None,
) -> str:
    """Concatenate structured data into a context string."""
    contexts = []
    if sub_community_reports:
        sub_community_reports = [
            report
            for report in sub_community_reports
            if schemas.COMMUNITY_ID in report
            and report[schemas.COMMUNITY_ID]
            and str(report[schemas.COMMUNITY_ID]).strip() != ""
        ]
        report_df = pd.DataFrame(sub_community_reports).drop_duplicates()
        if not report_df.empty:
            if report_df[schemas.COMMUNITY_ID].dtype == float:
                report_df[schemas.COMMUNITY_ID] = report_df[
                    schemas.COMMUNITY_ID
                ].astype(int)
            report_string = (
                f"----REPORTS-----\n{report_df.to_csv(index=False, sep=',')}"
            )
            contexts.append(report_string)


    if transition_records:
        transition_records = [
            transition
            for transition in transition_records
            if transition.get("source") and transition.get("to_target")
        ]
        transition_df = pd.DataFrame(transition_records).drop_duplicates()
        if not transition_df.empty:
            transition_df.sort_values(
                by=[
                    "changed_at_turn_index",
                    "changed_at_timestamp",
                    "source",
                    "relation_slot",
                ],
                inplace=True,
                na_position="last",
            )
            transition_string = (
                f"-----RELATIONSHIP_TRANSITIONS-----\n{transition_df.to_csv(index=False, sep=',')}"
            )
            contexts.append(transition_string)

    text_units = [
        unit
        for unit in text_units
        if "id" in unit and unit["id"] and str(unit["id"]).strip() != ""
    ]
    text_units_df = pd.DataFrame(text_units).drop_duplicates()
    if not text_units_df.empty:
        if text_units_df["id"].dtype == float:
            text_units_df["id"] = text_units_df["id"].astype(int)
        text_unit_string = (
            f"-----SOURCES-----\n{text_units_df.to_csv(index=False, sep=',')}"
        )
        contexts.append(text_unit_string)

    return "\n\n".join(contexts)


def sort_context(
    local_context: list[dict],
    tokenizer: Tokenizer,
    sub_community_reports: list[dict] | None = None,
    max_context_tokens: int | None = None,
    transition_records: list[dict] | None = None,
) -> str:
    """Sort local context by importance first, then temporal order as tie-breaker."""
    sorted_transitions = sorted(
        transition_records or [],
        key=lambda item: (
            _sortable_int(item.get("changed_at_turn_index")),
            str(item.get("changed_at_timestamp") or "~"),
            str(item.get("source") or ""),
            str(item.get("relation_slot") or ""),
        ),
    )

    def _fit_transitions(base_units: list[dict]) -> list[dict]:
        if not sorted_transitions:
            return []
        if not max_context_tokens:
            return sorted_transitions
        selected: list[dict] = []
        for transition in sorted_transitions:
            candidate = selected + [transition]
            candidate_text = get_context_string(
                base_units,
                sub_community_reports,
                transition_records=candidate,
            )
            if tokenizer.num_tokens(candidate_text) > max_context_tokens:
                break
            selected = candidate
        return selected

    sorted_text_units = sorted(
        local_context,
        key=lambda x: (
            -float(x.get(schemas.ENTITY_DEGREE) or 0),
            _temporal_sort_key(x),
            str(x.get("id") or ""),
        ),
    )

    current_text_units = []
    context_string = ""
    for record in sorted_text_units:
        current_text_units.append(record)
        if max_context_tokens:
            fitted_transitions = _fit_transitions(current_text_units)
            new_context_string = get_context_string(
                current_text_units,
                sub_community_reports,
                transition_records=fitted_transitions,
            )
            if tokenizer.num_tokens(new_context_string) > max_context_tokens:
                break

            context_string = new_context_string

    if context_string == "":
        fitted_transitions = _fit_transitions(sorted_text_units)
        return get_context_string(
            sorted_text_units,
            sub_community_reports,
            transition_records=fitted_transitions,
        )

    return context_string
