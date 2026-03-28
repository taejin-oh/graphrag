# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""Algorithms to build context data for local search prompt."""

import logging
import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pandas as pd
from graphrag_llm.tokenizer import Tokenizer
from graphrag_vectors import VectorStore

from graphrag.data_model.community import Community
from graphrag.data_model.community_report import CommunityReport
from graphrag.data_model.covariate import Covariate
from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.data_model.text_unit import TextUnit
from graphrag.index.utils.temporal_trace import trace_event
from graphrag.query.context_builder.builders import ContextBuilderResult
from graphrag.query.context_builder.community_context import (
    build_community_context,
)
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
from graphrag.query.context_builder.entity_extraction import (
    EntityVectorStoreKey,
    map_query_to_entities,
)
from graphrag.query.context_builder.local_context import (
    build_covariates_context,
    build_entity_context,
    build_relationship_context,
    get_candidate_context,
)
from graphrag.query.context_builder.source_context import (
    build_text_unit_context,
    count_relationships,
)
from graphrag.query.input.retrieval.community_reports import (
    get_candidate_communities,
)
from graphrag.query.input.retrieval.text_units import get_candidate_text_units
from graphrag.query.structured_search.base import LocalContextBuilder
from graphrag.query.structured_search.local_search.community_selection import (
    select_community_reports,
)
from graphrag.tokenizer.get_tokenizer import get_tokenizer

if TYPE_CHECKING:
    from graphrag_llm.embedding import LLMEmbedding

logger = logging.getLogger(__name__)


class LocalSearchMixedContext(LocalContextBuilder):
    """Build data context for local search prompt combining community reports and entity/relationship/covariate tables."""

    def __init__(
        self,
        entities: list[Entity],
        entity_text_embeddings: VectorStore,
        text_embedder: "LLMEmbedding",
        text_units: list[TextUnit] | None = None,
        communities: list[Community] | None = None,
        community_reports: list[CommunityReport] | None = None,
        relationships: list[Relationship] | None = None,
        covariates: dict[str, list[Covariate]] | None = None,
        tokenizer: Tokenizer | None = None,
        embedding_vectorstore_key: str = EntityVectorStoreKey.ID,
    ):
        if community_reports is None:
            community_reports = []
        if relationships is None:
            relationships = []
        if covariates is None:
            covariates = {}
        if text_units is None:
            text_units = []
        if communities is None:
            communities = []
        self.entities = {entity.id: entity for entity in entities}
        self.community_reports = {
            community.community_id: community for community in community_reports
        }
        self.community_metadata = {
            str(community.short_id): community
            for community in communities
            if community.short_id
        }
        self.community_metadata_by_id = {str(community.id): community for community in communities}
        self.text_units = {unit.id: unit for unit in text_units}
        self.relationships = {
            relationship.id: relationship for relationship in relationships
        }
        self.covariates = covariates
        self.entity_text_embeddings = entity_text_embeddings
        self.text_embedder = text_embedder
        self.tokenizer = tokenizer or get_tokenizer()
        self.embedding_vectorstore_key = embedding_vectorstore_key

    def build_context(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        include_entity_names: list[str] | None = None,
        exclude_entity_names: list[str] | None = None,
        conversation_history_max_turns: int | None = 5,
        conversation_history_user_turns_only: bool = True,
        max_context_tokens: int = 8000,
        text_unit_prop: float = 0.5,
        community_prop: float = 0.25,
        top_k_mapped_entities: int = 10,
        top_k_relationships: int = 10,
        include_community_rank: bool = False,
        include_entity_rank: bool = False,
        rank_description: str = "number of relationships",
        include_relationship_weight: bool = False,
        relationship_ranking_attribute: str = "rank",
        return_candidate_context: bool = False,
        use_community_summary: bool = False,
        min_community_rank: int = 0,
        community_context_name: str = "Reports",
        column_delimiter: str = "|",
        experimental_context_mode: bool = False,
        experimental_community_policy: str = "flat_ranked",
        experimental_history_enabled: bool = False,
        experimental_covariate_enabled: bool = True,
        experimental_context_max_tokens: int | None = None,
        experimental_condition_id: str | None = None,
        experimental_log_context_payload: bool = True,
        experimental_policy_preserve_mode: str = "fallback",
        **kwargs: dict[str, Any],
    ) -> ContextBuilderResult:
        """
        Build data context for local search prompt.

        Build a context by combining community reports and entity/relationship/covariate tables, and text units using a predefined ratio set by summary_prop.
        """
        if include_entity_names is None:
            include_entity_names = []
        if exclude_entity_names is None:
            exclude_entity_names = []
        if community_prop + text_unit_prop > 1:
            value_error = (
                "The sum of community_prop and text_unit_prop should not exceed 1."
            )
            raise ValueError(value_error)

        original_query = query
        entity_retrieval_query = query
        # map user query to entities
        # if there is conversation history, attached the previous user questions to the current query
        if conversation_history and not experimental_context_mode:
            pre_user_questions = "\n".join(
                conversation_history.get_user_turns(conversation_history_max_turns)
            )
            entity_retrieval_query = f"{query}\n{pre_user_questions}"

        selected_entities = map_query_to_entities(
            query=entity_retrieval_query,
            text_embedding_vectorstore=self.entity_text_embeddings,
            text_embedder=self.text_embedder,
            all_entities_dict=self.entities,
            embedding_vectorstore_key=self.embedding_vectorstore_key,
            include_entity_names=include_entity_names,
            exclude_entity_names=exclude_entity_names,
            k=top_k_mapped_entities,
            oversample_scaler=2,
        )

        if experimental_context_mode:
            return self._build_experimental_context(
                query=original_query,
                selected_entities=selected_entities,
                top_k_mapped_entities=top_k_mapped_entities,
                conversation_history=conversation_history,
                max_context_tokens=experimental_context_max_tokens
                or max_context_tokens,
                include_community_rank=include_community_rank,
                min_community_rank=min_community_rank,
                community_context_name=community_context_name,
                column_delimiter=column_delimiter,
                community_policy=experimental_community_policy,
                history_enabled=experimental_history_enabled,
                covariate_enabled=experimental_covariate_enabled,
                condition_id=experimental_condition_id,
                log_payload=experimental_log_context_payload,
                policy_preserve_mode=experimental_policy_preserve_mode,
            )

        # build context
        final_context = list[str]()
        final_context_data = dict[str, pd.DataFrame]()

        if conversation_history:
            # build conversation history context
            (
                conversation_history_context,
                conversation_history_context_data,
            ) = conversation_history.build_context(
                include_user_turns_only=conversation_history_user_turns_only,
                max_qa_turns=conversation_history_max_turns,
                column_delimiter=column_delimiter,
                max_context_tokens=max_context_tokens,
                recency_bias=False,
            )
            if conversation_history_context.strip() != "":
                final_context.append(conversation_history_context)
                final_context_data = conversation_history_context_data
                max_context_tokens = max_context_tokens - len(
                    self.tokenizer.encode(conversation_history_context)
                )

        # build community context
        community_tokens = max(int(max_context_tokens * community_prop), 0)
        community_context, community_context_data = self._build_community_context(
            selected_entities=selected_entities,
            max_context_tokens=community_tokens,
            use_community_summary=use_community_summary,
            column_delimiter=column_delimiter,
            include_community_rank=include_community_rank,
            min_community_rank=min_community_rank,
            return_candidate_context=return_candidate_context,
            context_name=community_context_name,
        )
        if community_context.strip() != "":
            final_context.append(community_context)
            final_context_data = {**final_context_data, **community_context_data}

        # build local (i.e. entity-relationship-covariate) context
        local_prop = 1 - community_prop - text_unit_prop
        local_tokens = max(int(max_context_tokens * local_prop), 0)
        local_context, local_context_data = self._build_local_context(
            selected_entities=selected_entities,
            max_context_tokens=local_tokens,
            include_entity_rank=include_entity_rank,
            rank_description=rank_description,
            include_relationship_weight=include_relationship_weight,
            top_k_relationships=top_k_relationships,
            relationship_ranking_attribute=relationship_ranking_attribute,
            return_candidate_context=return_candidate_context,
            column_delimiter=column_delimiter,
        )
        if local_context.strip() != "":
            final_context.append(str(local_context))
            final_context_data = {**final_context_data, **local_context_data}

        text_unit_tokens = max(int(max_context_tokens * text_unit_prop), 0)
        text_unit_context, text_unit_context_data = self._build_text_unit_context(
            selected_entities=selected_entities,
            max_context_tokens=text_unit_tokens,
            return_candidate_context=return_candidate_context,
        )

        if text_unit_context.strip() != "":
            final_context.append(text_unit_context)
            final_context_data = {**final_context_data, **text_unit_context_data}

        return ContextBuilderResult(
            context_chunks="\n\n".join(final_context),
            context_records=final_context_data,
        )

    def _build_experimental_context(
        self,
        *,
        query: str,
        selected_entities: list[Entity],
        top_k_mapped_entities: int,
        conversation_history: ConversationHistory | None,
        max_context_tokens: int,
        include_community_rank: bool,
        min_community_rank: int,
        community_context_name: str,
        column_delimiter: str,
        community_policy: str,
        history_enabled: bool,
        covariate_enabled: bool,
        condition_id: str | None,
        log_payload: bool,
        policy_preserve_mode: str,
    ) -> ContextBuilderResult:
        warnings: list[str] = []
        final_context: list[str] = []
        final_context_data: dict[str, pd.DataFrame] = {}

        ranked_matched_reports, ranked_all_reports = self._rank_community_reports(
            selected_entities=selected_entities
        )

        selection_result = select_community_reports(
            policy=community_policy,
            ranked_matched_reports=ranked_matched_reports,
            ranked_all_reports=ranked_all_reports,
            communities_by_short_id=self._community_metadata_view(),
            report_by_community_id=self.community_reports,
            max_tokens=max_context_tokens,
            token_counter=lambda report: self._community_report_token_cost(
                report=report,
                include_community_rank=include_community_rank,
                column_delimiter=column_delimiter,
            ),
            preserve_mode=policy_preserve_mode,
        )
        warnings.extend(selection_result.warnings)

        community_context, community_context_data = self._build_summary_only_community_context(
            selected_reports=selection_result.selected_reports,
            max_context_tokens=max_context_tokens,
            column_delimiter=column_delimiter,
            include_community_rank=include_community_rank,
            min_community_rank=min_community_rank,
            context_name=community_context_name,
        )
        community_tokens = len(self.tokenizer.encode(community_context))
        remaining_tokens = max(max_context_tokens - community_tokens, 0)
        if community_context.strip():
            final_context.append(community_context)
            final_context_data = {**final_context_data, **community_context_data}
        inserted_community_ids = self._extract_inserted_community_ids(
            context_data=community_context_data,
            context_name=community_context_name,
            selected_reports=selection_result.selected_reports,
        )

        covariate_context = ""
        covariate_tokens = 0
        covariate_records: dict[str, pd.DataFrame] = {}
        if covariate_enabled:
            covariate_context, covariate_records = self._build_covariate_only_context(
                selected_entities=selected_entities,
                max_context_tokens=remaining_tokens,
                column_delimiter=column_delimiter,
            )
            covariate_tokens = len(self.tokenizer.encode(covariate_context))
            remaining_tokens = max(remaining_tokens - covariate_tokens, 0)
            if covariate_context.strip():
                final_context.append(covariate_context)
                final_context_data = {**final_context_data, **covariate_records}

        history_context = ""
        history_tokens = 0
        if history_enabled and conversation_history:
            (
                history_context,
                history_tokens,
                history_records,
                history_warning,
            ) = self._build_history_block_if_fit(
                conversation_history=conversation_history,
                column_delimiter=column_delimiter,
                max_context_tokens=remaining_tokens,
            )
            if history_warning:
                warnings.append(history_warning)
            if history_context.strip():
                final_context.append(history_context)
                final_context_data = {**final_context_data, **history_records}
        elif history_enabled:
            warnings.append("History enabled but no conversation history was provided.")

        assembled_context = "\n\n".join(chunk for chunk in final_context if chunk.strip())
        assembled_context_tokens = len(self.tokenizer.encode(assembled_context))
        if assembled_context_tokens > max_context_tokens:
            warnings.append(
                "Assembled context exceeded max token budget; this indicates a builder bug."
            )

        condition_label = condition_id or (
            f"{community_policy}|h{int(history_enabled)}|c{int(covariate_enabled)}"
        )
        selected_pre_filter_ids = [
            report.community_id for report in selection_result.selected_reports
        ]
        dropped_community_ids = [
            community_id
            for community_id in selected_pre_filter_ids
            if community_id not in inserted_community_ids
        ]
        selected_communities_by_level = self._group_community_ids_by_level(
            inserted_community_ids
        )
        payload = {
            "condition_id": condition_label,
            "community_policy": community_policy,
            "policy_preserve_mode": policy_preserve_mode,
            "history_enabled": history_enabled,
            "covariate_enabled": covariate_enabled,
            "query": query,
            "mapped_entities_top_k": top_k_mapped_entities,
            "mapped_entities_count": len(selected_entities),
            "mapped_entity_titles": [entity.title for entity in selected_entities],
            "selected_community_ids": inserted_community_ids,
            "selected_community_levels": selected_communities_by_level,
            "selected_community_ids_pre_filter": selected_pre_filter_ids,
            "dropped_community_ids": dropped_community_ids,
            "primary_selected_ids": selection_result.primary_selected_ids,
            "fallback_selected_ids": selection_result.fallback_selected_ids,
            "community_summary_used": bool(community_context.strip()),
            "assembled_context": assembled_context,
            "community_tokens": community_tokens,
            "covariate_tokens": covariate_tokens,
            "history_tokens": history_tokens,
            "assembled_context_tokens": assembled_context_tokens,
            "max_context_tokens": max_context_tokens,
            "warnings": warnings,
        }
        if log_payload:
            logger.info(
                "[LOCAL_CONTEXT_PAYLOAD] %s",
                json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True),
            )
        trace_event(
            logger,
            stage="query_context",
            event="experimental_assembled_context",
            condition_id=condition_label,
            assembled_context_tokens=assembled_context_tokens,
            max_context_tokens=max_context_tokens,
            warnings_count=len(warnings),
            selected_communities=len(selection_result.selected_reports),
        )

        final_context_data["experimental_context"] = pd.DataFrame([payload])
        return ContextBuilderResult(
            context_chunks=assembled_context,
            context_records=final_context_data,
        )

    def _group_community_ids_by_level(
        self, community_ids: list[str]
    ) -> dict[str, list[str]]:
        by_level: dict[str, list[str]] = {}
        community_map = self._community_metadata_view()
        for community_id in community_ids:
            community = community_map.get(community_id)
            level = str(community.level) if community else "unknown"
            by_level.setdefault(level, [])
            by_level[level].append(community_id)
        return by_level

    def _rank_community_reports(
        self, selected_entities: list[Entity]
    ) -> tuple[list[CommunityReport], list[CommunityReport]]:
        community_matches: dict[str, int] = {}
        for entity in selected_entities:
            if entity.community_ids:
                for community_id in entity.community_ids:
                    community_matches[community_id] = (
                        community_matches.get(community_id, 0) + 1
                    )

        ranked_all_reports = list(self.community_reports.values())
        ranked_all_reports.sort(
            key=lambda report: (
                community_matches.get(report.community_id, 0),
                report.rank if report.rank is not None else 0,
            ),
            reverse=True,
        )
        ranked_matched_reports = [
            report
            for report in ranked_all_reports
            if community_matches.get(report.community_id, 0) > 0
        ]
        return ranked_matched_reports, ranked_all_reports

    def _community_metadata_view(self) -> dict[str, Community]:
        metadata = dict(self.community_metadata)
        for community_id, community in self.community_metadata_by_id.items():
            metadata.setdefault(community_id, community)
        return metadata

    def _build_summary_only_community_context(
        self,
        *,
        selected_reports: list[CommunityReport],
        max_context_tokens: int,
        column_delimiter: str,
        include_community_rank: bool,
        min_community_rank: int,
        context_name: str,
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        if not selected_reports:
            return ("", {context_name.lower(): pd.DataFrame()})

        context_text, context_data = build_community_context(
            community_reports=selected_reports,
            tokenizer=self.tokenizer,
            use_community_summary=True,
            column_delimiter=column_delimiter,
            shuffle_data=False,
            include_community_rank=include_community_rank,
            min_community_rank=min_community_rank,
            include_community_weight=False,
            max_context_tokens=max_context_tokens,
            single_batch=True,
            context_name=context_name,
        )
        if isinstance(context_text, list):
            context_text = "\n\n".join(context_text)
        return str(context_text), context_data

    def _extract_inserted_community_ids(
        self,
        *,
        context_data: dict[str, pd.DataFrame],
        context_name: str,
        selected_reports: list[CommunityReport],
    ) -> list[str]:
        context_key = context_name.lower()
        if context_key not in context_data:
            return []
        context_df = context_data[context_key]
        if "id" not in context_df.columns:
            return []
        short_id_to_community_id = {
            str(report.short_id): report.community_id
            for report in selected_reports
            if report.short_id
        }
        inserted_ids: list[str] = []
        for short_id in context_df["id"].astype(str).tolist():
            mapped = short_id_to_community_id.get(short_id, short_id)
            if mapped not in inserted_ids:
                inserted_ids.append(mapped)
        return inserted_ids

    def _community_report_token_cost(
        self,
        *,
        report: CommunityReport,
        include_community_rank: bool,
        column_delimiter: str,
    ) -> int:
        preview_context, _ = self._build_summary_only_community_context(
            selected_reports=[report],
            max_context_tokens=10**9,
            column_delimiter=column_delimiter,
            include_community_rank=include_community_rank,
            min_community_rank=0,
            context_name="Reports",
        )
        return len(self.tokenizer.encode(preview_context))

    def _build_covariate_only_context(
        self,
        *,
        selected_entities: list[Entity],
        max_context_tokens: int,
        column_delimiter: str,
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        if max_context_tokens <= 0:
            return "", {}

        context_chunks: list[str] = []
        context_records: dict[str, pd.DataFrame] = {}
        used_tokens = 0
        for covariate_name in self.covariates:
            context_text, context_df = build_covariates_context(
                selected_entities=selected_entities,
                covariates=self.covariates[covariate_name],
                tokenizer=self.tokenizer,
                max_context_tokens=max(max_context_tokens - used_tokens, 0),
                column_delimiter=column_delimiter,
                context_name=covariate_name,
            )
            if context_text.strip():
                new_tokens = len(self.tokenizer.encode(context_text))
                if used_tokens + new_tokens > max_context_tokens:
                    break
                used_tokens += new_tokens
                context_chunks.append(context_text)
            context_records[covariate_name.lower()] = context_df
        return "\n\n".join(context_chunks), context_records

    def _build_history_block_if_fit(
        self,
        *,
        conversation_history: ConversationHistory,
        column_delimiter: str,
        max_context_tokens: int,
    ) -> tuple[str, int, dict[str, pd.DataFrame], str | None]:
        full_context, history_data = conversation_history.build_context(
            tokenizer=self.tokenizer,
            include_user_turns_only=False,
            max_qa_turns=3,
            column_delimiter=column_delimiter,
            max_context_tokens=10**9,
            recency_bias=False,
        )
        history_tokens = len(self.tokenizer.encode(full_context))
        if history_tokens > max_context_tokens:
            return (
                "",
                0,
                {},
                "Conversation history block excluded because recent 3 turns exceeded remaining token budget.",
            )
        return full_context, history_tokens, history_data, None

    def _build_community_context(
        self,
        selected_entities: list[Entity],
        max_context_tokens: int = 4000,
        use_community_summary: bool = False,
        column_delimiter: str = "|",
        include_community_rank: bool = False,
        min_community_rank: int = 0,
        return_candidate_context: bool = False,
        context_name: str = "Reports",
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        """Add community data to the context window until it hits the max_context_tokens limit."""
        if len(selected_entities) == 0 or len(self.community_reports) == 0:
            return ("", {context_name.lower(): pd.DataFrame()})

        community_matches = {}
        for entity in selected_entities:
            # increase count of the community that this entity belongs to
            if entity.community_ids:
                for community_id in entity.community_ids:
                    community_matches[community_id] = (
                        community_matches.get(community_id, 0) + 1
                    )

        # sort communities by number of matched entities and rank
        selected_communities = [
            self.community_reports[community_id]
            for community_id in community_matches
            if community_id in self.community_reports
        ]
        for community in selected_communities:
            if community.attributes is None:
                community.attributes = {}
            community.attributes["matches"] = community_matches[community.community_id]
        selected_communities.sort(
            key=lambda x: (x.attributes["matches"], x.rank),  # type: ignore
            reverse=True,  # type: ignore
        )
        for community in selected_communities:
            del community.attributes["matches"]  # type: ignore

        context_text, context_data = build_community_context(
            community_reports=selected_communities,
            tokenizer=self.tokenizer,
            use_community_summary=use_community_summary,
            column_delimiter=column_delimiter,
            shuffle_data=False,
            include_community_rank=include_community_rank,
            min_community_rank=min_community_rank,
            max_context_tokens=max_context_tokens,
            single_batch=True,
            context_name=context_name,
        )
        if isinstance(context_text, list) and len(context_text) > 0:
            context_text = "\n\n".join(context_text)

        if return_candidate_context:
            candidate_context_data = get_candidate_communities(
                selected_entities=selected_entities,
                community_reports=list(self.community_reports.values()),
                use_community_summary=use_community_summary,
                include_community_rank=include_community_rank,
            )
            context_key = context_name.lower()
            if context_key not in context_data:
                context_data[context_key] = candidate_context_data
                context_data[context_key]["in_context"] = False
            else:
                if (
                    "id" in candidate_context_data.columns
                    and "id" in context_data[context_key].columns
                ):
                    candidate_context_data["in_context"] = candidate_context_data[
                        "id"
                    ].isin(  # cspell:disable-line
                        context_data[context_key]["id"]
                    )
                    context_data[context_key] = candidate_context_data
                else:
                    context_data[context_key]["in_context"] = True
        return (str(context_text), context_data)

    def _build_text_unit_context(
        self,
        selected_entities: list[Entity],
        max_context_tokens: int = 8000,
        return_candidate_context: bool = False,
        column_delimiter: str = "|",
        context_name: str = "Sources",
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        """Rank matching text units and add them to the context window until it hits the max_context_tokens limit."""
        if not selected_entities or not self.text_units:
            return ("", {context_name.lower(): pd.DataFrame()})
        selected_text_units = []
        text_unit_ids_set = set()

        unit_info_list = []
        relationship_values = list(self.relationships.values())

        for index, entity in enumerate(selected_entities):
            # get matching relationships
            entity_relationships = [
                rel
                for rel in relationship_values
                if rel.source == entity.title or rel.target == entity.title
            ]

            for text_id in entity.text_unit_ids or []:
                if text_id not in text_unit_ids_set and text_id in self.text_units:
                    selected_unit = deepcopy(self.text_units[text_id])
                    num_relationships = count_relationships(
                        entity_relationships, selected_unit
                    )
                    text_unit_ids_set.add(text_id)
                    unit_info_list.append((selected_unit, index, num_relationships))

        # sort by entity_order and the number of relationships desc
        unit_info_list.sort(key=lambda x: (x[1], -x[2]))

        selected_text_units = [unit[0] for unit in unit_info_list]
        selected_text_units = self._sort_text_units_by_temporal(selected_text_units)
        current_units, history_units = self._split_current_and_history(selected_text_units)
        trace_event(
            logger,
            stage="query_context",
            event="sources_split_current_history",
            selected_text_units=len(selected_text_units),
            current_units=len(current_units),
            history_units=len(history_units),
            split_rule="max(end_turn_index) == CURRENT",
        )

        current_text, current_data = build_text_unit_context(
            text_units=current_units or selected_text_units,
            tokenizer=self.tokenizer,
            max_context_tokens=max_context_tokens // 2 if history_units else max_context_tokens,
            shuffle_data=False,
            context_name=f"{context_name}_CURRENT",
            column_delimiter=column_delimiter,
        )
        history_text, history_data = ("", {})
        if history_units:
            history_text, history_data = build_text_unit_context(
                text_units=history_units,
                tokenizer=self.tokenizer,
                max_context_tokens=max_context_tokens // 2,
                shuffle_data=False,
                context_name=f"{context_name}_TIMELINE",
                column_delimiter=column_delimiter,
            )

        context_text = "\n\n".join([text for text in [current_text, history_text] if text.strip()])
        context_data = {**current_data, **history_data}
        trace_event(
            logger,
            stage="query_context",
            event="sources_context_built",
            context_keys=list(context_data.keys()),
            preview_fields={"context_text_preview": context_text},
        )

        if return_candidate_context:
            candidate_context_data = get_candidate_text_units(
                selected_entities=selected_entities,
                text_units=list(self.text_units.values()),
            )
            context_key = context_name.lower()
            context_variant_keys = [
                key
                for key in [
                    context_key,
                    f"{context_key}_current",
                    f"{context_key}_timeline",
                ]
                if key in context_data
            ]
            if not context_variant_keys:
                candidate_context_data["in_context"] = False
                context_data[context_key] = candidate_context_data
            else:
                if "id" in candidate_context_data.columns:
                    in_context_ids: set[str] = set()
                    for key in context_variant_keys:
                        if "id" in context_data[key].columns:
                            in_context_ids.update(
                                str(value) for value in context_data[key]["id"].tolist()
                            )
                    if in_context_ids:
                        candidate_context_data["in_context"] = candidate_context_data[
                            "id"
                        ].astype(str).isin(in_context_ids)
                    else:
                        candidate_context_data["in_context"] = False
                else:
                    candidate_context_data["in_context"] = False
                context_data[context_key] = candidate_context_data

        return (str(context_text), context_data)

    def _sort_text_units_by_temporal(self, text_units: list[TextUnit]) -> list[TextUnit]:
        def _safe_int(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return -1

        return sorted(
            text_units,
            key=lambda unit: (
                _safe_int((unit.attributes or {}).get("start_turn_index")),
                _safe_int((unit.attributes or {}).get("chunk_index_in_conversation")),
                str(unit.id),
            ),
        )

    def _split_current_and_history(
        self, text_units: list[TextUnit]
    ) -> tuple[list[TextUnit], list[TextUnit]]:
        if not text_units:
            return [], []

        def _safe_int(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return -1

        max_end_turn = max(
            _safe_int((unit.attributes or {}).get("end_turn_index")) for unit in text_units
        )
        if max_end_turn < 0:
            return text_units, []

        current = [
            unit
            for unit in text_units
            if _safe_int((unit.attributes or {}).get("end_turn_index")) == max_end_turn
        ]
        history = [
            unit
            for unit in text_units
            if _safe_int((unit.attributes or {}).get("end_turn_index")) < max_end_turn
        ]
        return current, history

    def _build_local_context(
        self,
        selected_entities: list[Entity],
        max_context_tokens: int = 8000,
        include_entity_rank: bool = False,
        rank_description: str = "relationship count",
        include_relationship_weight: bool = False,
        top_k_relationships: int = 10,
        relationship_ranking_attribute: str = "rank",
        return_candidate_context: bool = False,
        column_delimiter: str = "|",
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        """Build data context for local search prompt combining entity/relationship/covariate tables."""
        # build entity context
        entity_context, entity_context_data = build_entity_context(
            selected_entities=selected_entities,
            tokenizer=self.tokenizer,
            max_context_tokens=max_context_tokens,
            column_delimiter=column_delimiter,
            include_entity_rank=include_entity_rank,
            rank_description=rank_description,
            context_name="Entities",
        )
        entity_tokens = len(self.tokenizer.encode(entity_context))

        # build relationship-covariate context
        added_entities = []
        final_context = []
        final_context_data = {}

        # gradually add entities and associated metadata to the context until we reach limit
        for entity in selected_entities:
            current_context = []
            current_context_data = {}
            added_entities.append(entity)

            # build relationship context
            (
                relationship_context,
                relationship_context_data,
            ) = build_relationship_context(
                selected_entities=added_entities,
                relationships=list(self.relationships.values()),
                tokenizer=self.tokenizer,
                max_context_tokens=max_context_tokens,
                column_delimiter=column_delimiter,
                top_k_relationships=top_k_relationships,
                include_relationship_weight=include_relationship_weight,
                relationship_ranking_attribute=relationship_ranking_attribute,
                context_name="Relationships",
            )
            current_context.append(relationship_context)
            current_context_data["relationships"] = relationship_context_data
            total_tokens = entity_tokens + len(
                self.tokenizer.encode(relationship_context)
            )

            # build covariate context
            for covariate in self.covariates:
                covariate_context, covariate_context_data = build_covariates_context(
                    selected_entities=added_entities,
                    covariates=self.covariates[covariate],
                    tokenizer=self.tokenizer,
                    max_context_tokens=max_context_tokens,
                    column_delimiter=column_delimiter,
                    context_name=covariate,
                )
                total_tokens += len(self.tokenizer.encode(covariate_context))
                current_context.append(covariate_context)
                current_context_data[covariate.lower()] = covariate_context_data

            if total_tokens > max_context_tokens:
                logger.warning(
                    "Reached token limit - reverting to previous context state"
                )
                break

            final_context = current_context
            final_context_data = current_context_data

        # attach entity context to final context
        final_context_text = entity_context + "\n\n" + "\n\n".join(final_context)
        final_context_data["entities"] = entity_context_data

        if return_candidate_context:
            # we return all the candidate entities/relationships/covariates (not only those that were fitted into the context window)
            # and add a tag to indicate which records were included in the context window
            candidate_context_data = get_candidate_context(
                selected_entities=selected_entities,
                entities=list(self.entities.values()),
                relationships=list(self.relationships.values()),
                covariates=self.covariates,
                include_entity_rank=include_entity_rank,
                entity_rank_description=rank_description,
                include_relationship_weight=include_relationship_weight,
            )
            for key in candidate_context_data:
                candidate_df = candidate_context_data[key]
                if key not in final_context_data:
                    final_context_data[key] = candidate_df
                    final_context_data[key]["in_context"] = False
                else:
                    in_context_df = final_context_data[key]

                    if "id" in in_context_df.columns and "id" in candidate_df.columns:
                        candidate_df["in_context"] = candidate_df[
                            "id"
                        ].isin(  # cspell:disable-line
                            in_context_df["id"]
                        )
                        final_context_data[key] = candidate_df
                    else:
                        final_context_data[key]["in_context"] = True
        else:
            for key in final_context_data:
                final_context_data[key]["in_context"] = True
        return (final_context_text, final_context_data)
