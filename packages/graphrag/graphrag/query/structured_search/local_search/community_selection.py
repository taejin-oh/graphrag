# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Community selection policies for experimental local-search context assembly."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from graphrag.data_model.community import Community
from graphrag.data_model.community_report import CommunityReport


CommunityTokenCounter = Callable[[CommunityReport], int]


@dataclass
class CommunitySelectionResult:
    """Selection result for a community policy."""

    selected_reports: list[CommunityReport]
    warnings: list[str]


def select_community_reports(
    *,
    policy: str,
    ranked_matched_reports: list[CommunityReport],
    ranked_all_reports: list[CommunityReport],
    communities_by_short_id: dict[str, Community],
    report_by_community_id: dict[str, CommunityReport],
    max_tokens: int,
    token_counter: CommunityTokenCounter,
) -> CommunitySelectionResult:
    """Select community reports according to the configured policy."""
    if policy == "leaf_only":
        return _select_leaf_only(
            ranked_matched_reports=ranked_matched_reports,
            ranked_all_reports=ranked_all_reports,
            communities_by_short_id=communities_by_short_id,
            max_tokens=max_tokens,
            token_counter=token_counter,
        )
    if policy == "leaf_then_parent_mix":
        return _select_leaf_then_parent_mix(
            ranked_matched_reports=ranked_matched_reports,
            ranked_all_reports=ranked_all_reports,
            communities_by_short_id=communities_by_short_id,
            report_by_community_id=report_by_community_id,
            max_tokens=max_tokens,
            token_counter=token_counter,
        )
    if policy == "pyramid":
        return _select_pyramid(
            ranked_matched_reports=ranked_matched_reports,
            ranked_all_reports=ranked_all_reports,
            communities_by_short_id=communities_by_short_id,
            report_by_community_id=report_by_community_id,
            max_tokens=max_tokens,
            token_counter=token_counter,
        )
    if policy == "flat_ranked":
        return _select_flat_ranked(
            ranked_all_reports=ranked_all_reports,
            max_tokens=max_tokens,
            token_counter=token_counter,
        )
    return CommunitySelectionResult(
        selected_reports=[],
        warnings=[f"Unknown community selection policy '{policy}'."],
    )


def _select_leaf_only(
    *,
    ranked_matched_reports: list[CommunityReport],
    ranked_all_reports: list[CommunityReport],
    communities_by_short_id: dict[str, Community],
    max_tokens: int,
    token_counter: CommunityTokenCounter,
) -> CommunitySelectionResult:
    warnings: list[str] = []
    leaf_reports = [
        report
        for report in ranked_matched_reports
        if _is_leaf(report.community_id, communities_by_short_id)
    ]
    if not leaf_reports:
        warnings.append(
            "Policy leaf_only: no leaf communities found in matched candidates."
        )
    return _fill_with_fallback(
        primary=leaf_reports,
        fallback=ranked_all_reports,
        max_tokens=max_tokens,
        token_counter=token_counter,
        warnings=warnings,
    )


def _select_leaf_then_parent_mix(
    *,
    ranked_matched_reports: list[CommunityReport],
    ranked_all_reports: list[CommunityReport],
    communities_by_short_id: dict[str, Community],
    report_by_community_id: dict[str, CommunityReport],
    max_tokens: int,
    token_counter: CommunityTokenCounter,
) -> CommunitySelectionResult:
    warnings: list[str] = []
    leaf_reports = [
        report
        for report in ranked_matched_reports
        if _is_leaf(report.community_id, communities_by_short_id)
    ]
    if not leaf_reports:
        warnings.append(
            "Policy leaf_then_parent_mix: no leaf communities found in matched candidates."
        )

    parent_reports = _build_parent_candidates(
        leaf_reports=leaf_reports,
        ranked_all_reports=ranked_all_reports,
        communities_by_short_id=communities_by_short_id,
        report_by_community_id=report_by_community_id,
    )
    if not parent_reports:
        warnings.append(
            "Policy leaf_then_parent_mix: no parent communities available for leaf candidates."
        )

    selected: list[CommunityReport] = []
    selected_ids: set[str] = set()
    used_tokens = 0
    parent_added = False

    for report in leaf_reports:
        report_tokens = token_counter(report)
        if used_tokens + report_tokens > max_tokens:
            break
        selected.append(report)
        selected_ids.add(report.community_id)
        used_tokens += report_tokens

    for report in parent_reports:
        if report.community_id in selected_ids:
            continue
        report_tokens = token_counter(report)
        if used_tokens + report_tokens > max_tokens:
            continue
        selected.append(report)
        selected_ids.add(report.community_id)
        used_tokens += report_tokens
        parent_added = True

    if selected and not parent_added and parent_reports:
        replacement_leaf = selected[-1]
        candidate_parent = parent_reports[0]
        replacement_tokens = (
            used_tokens - token_counter(replacement_leaf) + token_counter(candidate_parent)
        )
        if replacement_tokens <= max_tokens:
            selected[-1] = candidate_parent
            selected_ids.discard(replacement_leaf.community_id)
            selected_ids.add(candidate_parent.community_id)
            used_tokens = replacement_tokens
        else:
            warnings.append(
                "Policy leaf_then_parent_mix: parent replacement skipped due to token limit."
            )

    if len(selected) == 1:
        warnings.append(
            "Policy leaf_then_parent_mix: only one community could be inserted."
        )

    selected, fallback_warnings = _append_fallback_reports(
        selected=selected,
        ranked_all_reports=ranked_all_reports,
        max_tokens=max_tokens,
        token_counter=token_counter,
    )
    warnings.extend(fallback_warnings)

    return CommunitySelectionResult(selected_reports=selected, warnings=warnings)


def _select_pyramid(
    *,
    ranked_matched_reports: list[CommunityReport],
    ranked_all_reports: list[CommunityReport],
    communities_by_short_id: dict[str, Community],
    report_by_community_id: dict[str, CommunityReport],
    max_tokens: int,
    token_counter: CommunityTokenCounter,
) -> CommunitySelectionResult:
    warnings: list[str] = []
    selected: list[CommunityReport] = []
    selected_ids: set[str] = set()
    used_tokens = 0

    leaf_reports = [
        report
        for report in ranked_matched_reports
        if _is_leaf(report.community_id, communities_by_short_id)
    ]
    if not leaf_reports:
        warnings.append("Policy pyramid: no leaf candidates found in matched reports.")
    else:
        top_leaf = leaf_reports[0]
        top_leaf_tokens = token_counter(top_leaf)
        if top_leaf_tokens <= max_tokens:
            selected.append(top_leaf)
            selected_ids.add(top_leaf.community_id)
            used_tokens += top_leaf_tokens
        else:
            warnings.append(
                "Policy pyramid: highest-priority leaf could not fit in token budget."
            )

    parent_candidates, grandparent_candidates = _build_hierarchy_candidates(
        leaf_reports=leaf_reports,
        ranked_all_reports=ranked_all_reports,
        communities_by_short_id=communities_by_short_id,
        report_by_community_id=report_by_community_id,
    )

    parent_added = False
    for report in parent_candidates:
        if report.community_id in selected_ids:
            continue
        report_tokens = token_counter(report)
        if used_tokens + report_tokens > max_tokens:
            continue
        selected.append(report)
        selected_ids.add(report.community_id)
        used_tokens += report_tokens
        parent_added = True

    if grandparent_candidates and not parent_added:
        warnings.append(
            "Policy pyramid: skipped grandparent insertion because no parent fit in token budget."
        )

    if parent_added:
        for report in grandparent_candidates:
            if report.community_id in selected_ids:
                continue
            report_tokens = token_counter(report)
            if used_tokens + report_tokens > max_tokens:
                continue
            selected.append(report)
            selected_ids.add(report.community_id)
            used_tokens += report_tokens

    selected, fallback_warnings = _append_fallback_reports(
        selected=selected,
        ranked_all_reports=ranked_all_reports,
        max_tokens=max_tokens,
        token_counter=token_counter,
    )
    warnings.extend(fallback_warnings)

    return CommunitySelectionResult(selected_reports=selected, warnings=warnings)


def _select_flat_ranked(
    *,
    ranked_all_reports: list[CommunityReport],
    max_tokens: int,
    token_counter: CommunityTokenCounter,
) -> CommunitySelectionResult:
    selected = []
    used_tokens = 0
    for report in ranked_all_reports:
        report_tokens = token_counter(report)
        if used_tokens + report_tokens > max_tokens:
            continue
        selected.append(report)
        used_tokens += report_tokens
    return CommunitySelectionResult(selected_reports=selected, warnings=[])


def _fill_with_fallback(
    *,
    primary: list[CommunityReport],
    fallback: list[CommunityReport],
    max_tokens: int,
    token_counter: CommunityTokenCounter,
    warnings: list[str],
) -> CommunitySelectionResult:
    selected = []
    selected_ids: set[str] = set()
    used_tokens = 0

    for report in primary:
        report_tokens = token_counter(report)
        if used_tokens + report_tokens > max_tokens:
            continue
        selected.append(report)
        selected_ids.add(report.community_id)
        used_tokens += report_tokens

    for report in fallback:
        if report.community_id in selected_ids:
            continue
        report_tokens = token_counter(report)
        if used_tokens + report_tokens > max_tokens:
            continue
        selected.append(report)
        selected_ids.add(report.community_id)
        used_tokens += report_tokens

    if not selected:
        warnings.append("No community reports could be selected under token budget.")
    return CommunitySelectionResult(selected_reports=selected, warnings=warnings)


def _append_fallback_reports(
    *,
    selected: list[CommunityReport],
    ranked_all_reports: list[CommunityReport],
    max_tokens: int,
    token_counter: CommunityTokenCounter,
) -> tuple[list[CommunityReport], list[str]]:
    warnings: list[str] = []
    selected_ids = {report.community_id for report in selected}
    used_tokens = sum(token_counter(report) for report in selected)
    for report in ranked_all_reports:
        if report.community_id in selected_ids:
            continue
        report_tokens = token_counter(report)
        if used_tokens + report_tokens > max_tokens:
            continue
        selected.append(report)
        selected_ids.add(report.community_id)
        used_tokens += report_tokens
    if not selected:
        warnings.append("No community reports could be selected under token budget.")
    return selected, warnings


def _build_parent_candidates(
    *,
    leaf_reports: list[CommunityReport],
    ranked_all_reports: list[CommunityReport],
    communities_by_short_id: dict[str, Community],
    report_by_community_id: dict[str, CommunityReport],
) -> list[CommunityReport]:
    parent_ids = []
    seen: set[str] = set()
    for leaf in leaf_reports:
        meta = communities_by_short_id.get(leaf.community_id)
        parent_id = str(meta.parent) if meta and meta.parent else ""
        if parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent_ids.append(parent_id)

    ranked_position = {
        report.community_id: idx for idx, report in enumerate(ranked_all_reports)
    }
    parent_reports = [
        report_by_community_id[parent_id]
        for parent_id in parent_ids
        if parent_id in report_by_community_id
    ]
    parent_reports.sort(
        key=lambda report: ranked_position.get(report.community_id, 10**9)
    )
    return parent_reports


def _build_hierarchy_candidates(
    *,
    leaf_reports: list[CommunityReport],
    ranked_all_reports: list[CommunityReport],
    communities_by_short_id: dict[str, Community],
    report_by_community_id: dict[str, CommunityReport],
) -> tuple[list[CommunityReport], list[CommunityReport]]:
    ranked_position = {
        report.community_id: idx for idx, report in enumerate(ranked_all_reports)
    }
    parent_ids: set[str] = set()
    grandparent_ids: set[str] = set()
    for leaf in leaf_reports:
        meta = communities_by_short_id.get(leaf.community_id)
        parent_id = str(meta.parent) if meta and meta.parent else ""
        if not parent_id:
            continue
        parent_ids.add(parent_id)
        parent_meta = communities_by_short_id.get(parent_id)
        grandparent_id = str(parent_meta.parent) if parent_meta and parent_meta.parent else ""
        if grandparent_id:
            grandparent_ids.add(grandparent_id)

    parents = [
        report_by_community_id[parent_id]
        for parent_id in parent_ids
        if parent_id in report_by_community_id
    ]
    grandparents = [
        report_by_community_id[grandparent_id]
        for grandparent_id in grandparent_ids
        if grandparent_id in report_by_community_id
    ]
    parents.sort(key=lambda report: ranked_position.get(report.community_id, 10**9))
    grandparents.sort(
        key=lambda report: ranked_position.get(report.community_id, 10**9)
    )
    return parents, grandparents


def _is_leaf(community_id: str, communities_by_short_id: dict[str, Community]) -> bool:
    meta = communities_by_short_id.get(community_id)
    if meta is None:
        return False
    return len(meta.children or []) == 0
