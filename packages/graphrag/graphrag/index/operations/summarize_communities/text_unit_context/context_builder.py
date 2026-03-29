# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Context builders for text units."""

import logging
from typing import cast

import pandas as pd
from graphrag_llm.tokenizer import Tokenizer

import graphrag.data_model.schemas as schemas
from graphrag.index.operations.summarize_communities.build_mixed_context import (
    build_mixed_context,
)
from graphrag.index.operations.summarize_communities.text_unit_context.prep_text_units import (
    prep_text_units,
)
from graphrag.index.operations.summarize_communities.text_unit_context.sort_context import (
    sort_context,
)

logger = logging.getLogger(__name__)


def _as_iterable_ids(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    text = str(value).strip()
    return [text] if text else []


def _build_transition_records(
    community_membership_df: pd.DataFrame,
    relationship_transitions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map relationship transitions to communities by text units and entities.

    We prioritize text-unit overlap (strongest evidence of community membership), and
    use source-entity overlap as a secondary fallback for rows that may miss text-unit
    links in incremental pipelines.
    """
    if relationship_transitions_df is None or relationship_transitions_df.empty:
        return pd.DataFrame(columns=[schemas.COMMUNITY_ID, "transition_records"])

    transition_df = relationship_transitions_df.copy()
    transition_df["changed_at_text_unit_id"] = transition_df[
        "changed_at_text_unit_id"
    ].astype(str)

    membership_rows: list[dict[str, int | str]] = []
    for row in community_membership_df.itertuples(index=False):
        community_id = int(getattr(row, schemas.COMMUNITY_ID))
        text_unit_ids = _as_iterable_ids(getattr(row, schemas.TEXT_UNIT_IDS, []))
        entity_ids = _as_iterable_ids(getattr(row, schemas.ENTITY_IDS, []))
        for tid in text_unit_ids:
            membership_rows.append(
                {
                    schemas.COMMUNITY_ID: community_id,
                    "membership_text_unit_id": str(tid),
                    "membership_entity_title": None,
                }
            )
        for entity in entity_ids:
            membership_rows.append(
                {
                    schemas.COMMUNITY_ID: community_id,
                    "membership_text_unit_id": None,
                    "membership_entity_title": str(entity),
                }
            )

    if not membership_rows:
        return pd.DataFrame(columns=[schemas.COMMUNITY_ID, "transition_records"])

    membership_df = pd.DataFrame(membership_rows)

    by_text = transition_df.merge(
        membership_df.dropna(subset=["membership_text_unit_id"]),
        left_on="changed_at_text_unit_id",
        right_on="membership_text_unit_id",
        how="inner",
    )
    by_entity = transition_df.merge(
        membership_df.dropna(subset=["membership_entity_title"]),
        left_on="source",
        right_on="membership_entity_title",
        how="inner",
    )

    candidate = pd.concat([by_text, by_entity], ignore_index=True).drop_duplicates(
        subset=[
            schemas.COMMUNITY_ID,
            "source",
            "relation_slot",
            "from_target",
            "to_target",
            "changed_at_text_unit_id",
            "conversation_id",
        ]
    )
    if candidate.empty:
        return pd.DataFrame(columns=[schemas.COMMUNITY_ID, "transition_records"])

    candidate = candidate.sort_values(
        by=[
            schemas.COMMUNITY_ID,
            "changed_at_turn_index",
            "changed_at_timestamp",
            "source",
            "relation_slot",
        ],
        na_position="last",
    )
    candidate["transition_records"] = candidate.apply(
        lambda x: {
            "record_type": "relationship_transition",
            "source": x.get("source"),
            "relation_slot": x.get("relation_slot"),
            "from_target": x.get("from_target"),
            "to_target": x.get("to_target"),
            "conversation_id": x.get("conversation_id"),
            "changed_at_turn_index": x.get("changed_at_turn_index"),
            "changed_at_timestamp": x.get("changed_at_timestamp"),
            "previous_text_unit_id": x.get("previous_text_unit_id"),
            "changed_at_text_unit_id": x.get("changed_at_text_unit_id"),
        },
        axis=1,
    )
    return (
        candidate.groupby(schemas.COMMUNITY_ID)["transition_records"]
        .agg(list)
        .reset_index()
    )


def build_local_context(
    community_membership_df: pd.DataFrame,
    text_units_df: pd.DataFrame,
    node_df: pd.DataFrame,
    tokenizer: Tokenizer,
    max_context_tokens: int = 16000,
    relationship_transitions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Prep context data for community report generation using text unit data.

    Community membership has columns [COMMUNITY_ID, COMMUNITY_LEVEL, ENTITY_IDS, RELATIONSHIP_IDS, TEXT_UNIT_IDS]
    """
    # get text unit details, include short_id, text, and entity degree (sum of degrees of the text unit's nodes that belong to a community)
    prepped_text_units_df = prep_text_units(text_units_df, node_df)
    prepped_text_units_df = prepped_text_units_df.rename(
        columns={
            schemas.ID: schemas.TEXT_UNIT_IDS,
            schemas.COMMUNITY_ID: schemas.COMMUNITY_ID,
        }
    )

    # merge text unit details with community membership
    context_df = community_membership_df.loc[
        :, [schemas.COMMUNITY_ID, schemas.COMMUNITY_LEVEL, schemas.TEXT_UNIT_IDS]
    ]
    context_df = context_df.explode(schemas.TEXT_UNIT_IDS)
    context_df = context_df.merge(
        prepped_text_units_df,
        on=[schemas.TEXT_UNIT_IDS, schemas.COMMUNITY_ID],
        how="left",
    )

    context_df[schemas.ALL_CONTEXT] = context_df.apply(
        lambda x: {
            "id": x[schemas.ALL_DETAILS][schemas.SHORT_ID],
            "text": x[schemas.ALL_DETAILS][schemas.TEXT],
            "entity_degree": x[schemas.ALL_DETAILS][schemas.ENTITY_DEGREE],
            schemas.START_TURN_INDEX: x[schemas.ALL_DETAILS].get(
                schemas.START_TURN_INDEX
            ),
            schemas.END_TURN_INDEX: x[schemas.ALL_DETAILS].get(
                schemas.END_TURN_INDEX
            ),
            schemas.TURN_TIMESTAMP_START: x[schemas.ALL_DETAILS].get(
                schemas.TURN_TIMESTAMP_START
            ),
            schemas.TURN_TIMESTAMP_END: x[schemas.ALL_DETAILS].get(
                schemas.TURN_TIMESTAMP_END
            ),
            schemas.CHUNK_INDEX_IN_CONVERSATION: x[schemas.ALL_DETAILS].get(
                schemas.CHUNK_INDEX_IN_CONVERSATION
            ),
        },
        axis=1,
    )

    context_df = (
        context_df
        .groupby([schemas.COMMUNITY_ID, schemas.COMMUNITY_LEVEL])
        .agg({schemas.ALL_CONTEXT: list})
        .reset_index()
    )

    transition_context_df = _build_transition_records(
        community_membership_df=community_membership_df,
        relationship_transitions_df=(
            relationship_transitions_df
            if relationship_transitions_df is not None
            else pd.DataFrame()
        ),
    )
    context_df = context_df.merge(transition_context_df, on=schemas.COMMUNITY_ID, how="left")
    context_df["transition_records"] = context_df["transition_records"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    context_df[schemas.CONTEXT_STRING] = context_df.apply(
        lambda row: sort_context(
            row[schemas.ALL_CONTEXT],
            tokenizer,
            transition_records=row["transition_records"],
        ),
        axis=1,
    )
    context_df[schemas.CONTEXT_SIZE] = context_df[schemas.CONTEXT_STRING].apply(
        lambda x: tokenizer.num_tokens(x)
    )
    context_df[schemas.CONTEXT_EXCEED_FLAG] = context_df[schemas.CONTEXT_SIZE].apply(
        lambda x: x > max_context_tokens
    )

    return context_df


def build_level_context(
    report_df: pd.DataFrame | None,
    community_hierarchy_df: pd.DataFrame,
    local_context_df: pd.DataFrame,
    level: int,
    tokenizer: Tokenizer,
    max_context_tokens: int = 16000,
) -> pd.DataFrame:
    """
    Prep context for each community in a given level.

    For each community:
    - Check if local context fits within the limit, if yes use local context
    - If local context exceeds the limit, iteratively replace local context with sub-community reports, starting from the biggest sub-community
    """
    if report_df is None or report_df.empty:
        # there is no report to substitute with, so we just trim the local context of the invalid context records
        # this case should only happen at the bottom level of the community hierarchy where there are no sub-communities
        level_context_df = local_context_df[
            local_context_df[schemas.COMMUNITY_LEVEL] == level
        ]

        valid_context_df = cast(
            "pd.DataFrame",
            level_context_df[~level_context_df[schemas.CONTEXT_EXCEED_FLAG]],
        )
        invalid_context_df = cast(
            "pd.DataFrame",
            level_context_df[level_context_df[schemas.CONTEXT_EXCEED_FLAG]],
        )

        if invalid_context_df.empty:
            return valid_context_df

        invalid_context_df.loc[:, [schemas.CONTEXT_STRING]] = invalid_context_df.apply(
            lambda row: sort_context(
                row[schemas.ALL_CONTEXT],
                tokenizer,
                max_context_tokens=max_context_tokens,
                transition_records=row.get("transition_records", []),
            ),
            axis=1,
        )
        invalid_context_df.loc[:, [schemas.CONTEXT_SIZE]] = invalid_context_df[
            schemas.CONTEXT_STRING
        ].apply(lambda x: tokenizer.num_tokens(x))
        invalid_context_df.loc[:, [schemas.CONTEXT_EXCEED_FLAG]] = False

        return pd.concat([valid_context_df, invalid_context_df])

    level_context_df = local_context_df[
        local_context_df[schemas.COMMUNITY_LEVEL] == level
    ]

    # exclude those that already have reports
    level_context_df = level_context_df.merge(
        report_df[[schemas.COMMUNITY_ID]],
        on=schemas.COMMUNITY_ID,
        how="outer",
        indicator=True,
    )
    level_context_df = level_context_df[level_context_df["_merge"] == "left_only"].drop(
        "_merge", axis=1
    )
    valid_context_df = cast(
        "pd.DataFrame",
        level_context_df[level_context_df[schemas.CONTEXT_EXCEED_FLAG] is False],
    )
    invalid_context_df = cast(
        "pd.DataFrame",
        level_context_df[level_context_df[schemas.CONTEXT_EXCEED_FLAG] is True],
    )

    if invalid_context_df.empty:
        return valid_context_df

    # for each invalid context, we will try to substitute with sub-community reports
    # first get local context and report (if available) for each sub-community
    sub_report_df = report_df[report_df[schemas.COMMUNITY_LEVEL] == level + 1].drop(
        [schemas.COMMUNITY_LEVEL], axis=1
    )
    sub_context_df = local_context_df[
        local_context_df[schemas.COMMUNITY_LEVEL] == level + 1
    ]
    sub_context_df = sub_context_df.merge(
        sub_report_df, on=schemas.COMMUNITY_ID, how="left"
    )
    sub_context_df.rename(
        columns={schemas.COMMUNITY_ID: schemas.SUB_COMMUNITY}, inplace=True
    )

    # collect all sub communities' contexts for each community
    community_df = community_hierarchy_df[
        community_hierarchy_df[schemas.COMMUNITY_LEVEL] == level
    ].drop([schemas.COMMUNITY_LEVEL], axis=1)
    community_df = community_df.merge(
        invalid_context_df[[schemas.COMMUNITY_ID]], on=schemas.COMMUNITY_ID, how="inner"
    )
    community_df = community_df.merge(
        sub_context_df[
            [
                schemas.SUB_COMMUNITY,
                schemas.FULL_CONTENT,
                schemas.ALL_CONTEXT,
                schemas.CONTEXT_SIZE,
            ]
        ],
        on=schemas.SUB_COMMUNITY,
        how="left",
    )
    community_df[schemas.ALL_CONTEXT] = community_df.apply(
        lambda x: {
            schemas.SUB_COMMUNITY: x[schemas.SUB_COMMUNITY],
            schemas.ALL_CONTEXT: x[schemas.ALL_CONTEXT],
            schemas.FULL_CONTENT: x[schemas.FULL_CONTENT],
            schemas.CONTEXT_SIZE: x[schemas.CONTEXT_SIZE],
        },
        axis=1,
    )
    community_df = (
        community_df
        .groupby(schemas.COMMUNITY_ID)
        .agg({schemas.ALL_CONTEXT: list})
        .reset_index()
    )
    community_df[schemas.CONTEXT_STRING] = community_df[schemas.ALL_CONTEXT].apply(
        lambda x: build_mixed_context(x, tokenizer, max_context_tokens)
    )
    community_df[schemas.CONTEXT_SIZE] = community_df[schemas.CONTEXT_STRING].apply(
        lambda x: tokenizer.num_tokens(x)
    )
    community_df[schemas.CONTEXT_EXCEED_FLAG] = False
    community_df[schemas.COMMUNITY_LEVEL] = level

    # handle any remaining invalid records that can't be subsituted with sub-community reports
    # this should be rare, but if it happens, we will just trim the local context to fit the limit
    remaining_df = invalid_context_df.merge(
        community_df[[schemas.COMMUNITY_ID]],
        on=schemas.COMMUNITY_ID,
        how="outer",
        indicator=True,
    )
    remaining_df = remaining_df[remaining_df["_merge"] == "left_only"].drop(
        "_merge", axis=1
    )
    remaining_df[schemas.CONTEXT_STRING] = cast(
        "pd.DataFrame", remaining_df[schemas.ALL_CONTEXT]
    ).apply(lambda x: sort_context(x, tokenizer, max_context_tokens=max_context_tokens))
    remaining_df[schemas.CONTEXT_SIZE] = cast(
        "pd.DataFrame", remaining_df[schemas.CONTEXT_STRING]
    ).apply(lambda x: tokenizer.num_tokens(x))
    remaining_df[schemas.CONTEXT_EXCEED_FLAG] = False

    return cast(
        "pd.DataFrame", pd.concat([valid_context_df, community_df, remaining_df])
    )
