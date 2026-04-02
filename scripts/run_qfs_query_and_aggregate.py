#!/usr/bin/env python3
"""Batch query runner + assembled_context aggregator for QFS inputs.

Usage examples:
  # 기본 실행: input_chat 전체를 순회해 policy x covariate 조건별 결과 생성
  python scripts/run_qfs_query_and_aggregate.py

  # 특정 test_case만 실행
  python scripts/run_qfs_query_and_aggregate.py --test-case case_a

  # 특정 test_id만 골라 실행
  python scripts/run_qfs_query_and_aggregate.py --test-case case_a --test-ids 001 003

  # 디버그 로그 + assembled_context 본문 출력
  python scripts/run_qfs_query_and_aggregate.py --test-case case_a --test-ids 001 --debug --show-assembled-context

  # dry-run
  python scripts/run_qfs_query_and_aggregate.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "packages" / "graphrag"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import graphrag.api as api
from graphrag.cli.query import _resolve_output_files
from graphrag.config.load_config import load_config
from graphrag.tokenizer.get_tokenizer import get_tokenizer

SUPPORTED_POLICIES = ["default"]

_DROP_COLUMNS = {"id", "title", "nid"}
_DATA_CITATION_RE = re.compile(r"\[Data:[^\]]+\]")
_HUMAN_KEEP_FIELDS = {"summary", "content", "current_state", "date_range", "entity", "timeline_events"}


def _iter_test_targets(input_chat_root: Path, test_case_filter: list[str] | None) -> list[tuple[str, str, Path]]:
    return _iter_test_targets_with_filter(
        input_chat_root=input_chat_root,
        test_case_filter=test_case_filter,
        test_id_filters=None,
    )


def _iter_test_targets_with_filter(
    *,
    input_chat_root: Path,
    test_case_filter: list[str] | None,
    test_id_filters: list[str] | None,
) -> list[tuple[str, str, Path]]:
    if not input_chat_root.exists():
        raise FileNotFoundError(f"input_chat root not found: {input_chat_root}")

    targets: list[tuple[str, str, Path]] = []

    case_dirs = {p.name: p for p in input_chat_root.iterdir() if p.is_dir()}
    if test_case_filter:
        missing_cases = [test_case for test_case in test_case_filter if test_case not in case_dirs]
        if missing_cases:
            raise FileNotFoundError(
                f"Requested test_case not found under input_chat root: {missing_cases}"
            )
        selected_cases = test_case_filter
    else:
        selected_cases = sorted(case_dirs.keys())

    for test_case in selected_cases:
        test_case_dir = case_dirs[test_case]
        id_dirs = {p.name: p for p in test_case_dir.iterdir() if p.is_dir()}
        if test_id_filters:
            missing_ids = [test_id for test_id in test_id_filters if test_id not in id_dirs]
            if missing_ids:
                raise FileNotFoundError(
                    f"Requested test_id not found under selected test_case '{test_case}': {missing_ids}"
                )
            selected_ids = test_id_filters
        else:
            selected_ids = sorted(id_dirs.keys())

        for test_id in selected_ids:
            test_id_dir = id_dirs[test_id]
            output_dir = test_id_dir / "output"
            probing_path = test_id_dir / "probing_questions" / "probing_questions.json"
            if not output_dir.exists():
                raise FileNotFoundError(
                    f"index output directory not found for {test_case}/{test_id}: {output_dir}"
                )
            if not probing_path.exists():
                raise FileNotFoundError(
                    f"probing_questions.json not found for {test_case}/{test_id}: {probing_path}"
                )
            targets.append((test_case, test_id, test_id_dir))

    if not targets:
        raise ValueError("No test targets found under input_chat.")
    return targets


def _load_questions(path: Path) -> dict[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid probing_questions format (dict expected): {path}")
    return data


def _condition_key(policy: str) -> str:
    return f"policy={policy}"


def _extract_selected_community_ids(context_data: dict[str, Any]) -> list[str]:
    reports_df = context_data.get("reports")
    if reports_df is None or not hasattr(reports_df, "columns"):
        return []
    id_column = "id" if "id" in reports_df.columns else None
    if id_column is None:
        for candidate in ["community", "community_id", "short_id", "human_readable_id"]:
            if candidate in reports_df.columns:
                id_column = candidate
                break
    if id_column is None:
        return []
    values = [str(v) for v in reports_df[id_column].tolist() if str(v).strip()]
    return list(dict.fromkeys(values))


def _extract_reports_header_column(assembled_context: str) -> str:
    lines = assembled_context.splitlines()
    in_reports = False
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if text.startswith("-----") and text.endswith("-----"):
            in_reports = text.strip("-").strip().lower() == "reports"
            continue
        if in_reports and "|" in text:
            headers = [h.strip().lower() for h in text.split("|")]
            if "summary" in headers:
                return "summary"
            if "content" in headers:
                return "content"
            return "unknown"
    return "unknown"


def _extract_reports_table_rows(assembled_context: str) -> list[dict[str, str]]:
    lines = assembled_context.splitlines()
    in_reports = False
    headers: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if text.startswith("-----") and text.endswith("-----"):
            in_reports = text.strip("-").strip().lower() == "reports"
            headers = None
            continue
        if not in_reports or "|" not in text:
            continue
        parts = [p.strip() for p in text.split("|")]
        if headers is None:
            headers = parts
            continue
        if len(parts) != len(headers):
            continue
        rows.append(
            {
                k.strip().lower().replace(" ", "_").replace("-", "_"): v.strip()
                for k, v in zip(headers, parts, strict=True)
            }
        )
    return rows


def _normalize_field_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _match_field_key(candidates: list[str], field_name: str) -> str | None:
    target = _normalize_field_name(field_name)
    if target in candidates:
        return target
    # fuzzy fallback
    if target == "current_state":
        for candidate in candidates:
            if "current" in candidate and "state" in candidate:
                return candidate
    if target == "timeline_events":
        for candidate in candidates:
            if "timeline" in candidate and ("event" in candidate or "events" in candidate):
                return candidate
    return None


def _extract_report_field_values(
    *,
    context_data: dict[str, Any],
    assembled_context: str,
    field_name: str,
) -> list[str]:
    values: list[str] = []
    reports_df = context_data.get("reports")
    if reports_df is not None and hasattr(reports_df, "columns"):
        normalized_to_actual = {
            _normalize_field_name(str(column)): column for column in reports_df.columns
        }
        actual_key = _match_field_key(list(normalized_to_actual.keys()), field_name)
        actual_column = normalized_to_actual.get(actual_key) if actual_key else None
        if actual_column is not None:
            values.extend(
                str(value).strip()
                for value in reports_df[actual_column].tolist()
                if str(value).strip()
            )
    if not values:
        for row in _extract_reports_table_rows(assembled_context):
            match_key = _match_field_key(list(row.keys()), field_name)
            value = row.get(match_key, "").strip() if match_key else ""
            if value:
                values.append(value)
    return list(dict.fromkeys(values))


def _extract_field_values_from_community_reports(
    *,
    community_reports_df: Any,
    selected_rows: list[dict[str, str]],
    field_name: str,
) -> list[str]:
    if community_reports_df is None or not hasattr(community_reports_df, "columns"):
        return []
    normalized_to_actual = {
        _normalize_field_name(str(column)): column for column in community_reports_df.columns
    }
    field_key = _match_field_key(list(normalized_to_actual.keys()), field_name)
    if not field_key:
        return []
    field_column = normalized_to_actual[field_key]
    id_columns = ["community", "community_id", "id", "short_id", "human_readable_id"]
    existing_id_columns = [c for c in id_columns if c in community_reports_df.columns]
    values: list[str] = []
    for item in selected_rows:
        matched = None
        if item["id"]:
            for id_col in existing_id_columns:
                probe = community_reports_df[
                    community_reports_df[id_col].astype(str) == item["id"]
                ]
                if not probe.empty:
                    matched = probe.iloc[0]
                    break
        if matched is None and item["title"] and "title" in community_reports_df.columns:
            probe = community_reports_df[
                community_reports_df["title"].astype(str) == item["title"]
            ]
            if not probe.empty:
                matched = probe.iloc[0]
        if matched is not None:
            value = str(matched.get(field_column, "")).strip()
            if value:
                values.append(value)
    return list(dict.fromkeys(values))


def _extract_selected_communities(
    *,
    context_data: dict[str, Any],
    community_reports_df: Any,
    assembled_context: str,
) -> list[dict[str, str]]:
    reports_df = context_data.get("reports")
    rows: list[dict[str, str]] = []
    if reports_df is not None and hasattr(reports_df, "iterrows"):
        id_candidates = ["id", "community", "community_id", "short_id", "human_readable_id"]
        summary_candidates = ["summary", "content", "full_content"]
        for _, row in reports_df.iterrows():
            rid = ""
            for col in id_candidates:
                value = row.get(col) if hasattr(row, "get") else None
                if value is not None and str(value).strip():
                    rid = str(value).strip()
                    break
            title = str(row.get("title", "")).strip() if hasattr(row, "get") else ""
            summary = ""
            for col in summary_candidates:
                value = row.get(col) if hasattr(row, "get") else None
                if value is not None and str(value).strip():
                    summary = str(value).strip()
                    break
            rows.append({"id": rid, "title": title, "summary": summary})

    # 보강: id/summary가 비어 있으면 community_reports 원본 테이블에서 title 기반으로 보완
    if community_reports_df is not None and hasattr(community_reports_df, "columns"):
        id_columns = ["community", "community_id", "id", "short_id", "human_readable_id"]
        existing_id_columns = [c for c in id_columns if c in community_reports_df.columns]
        for item in rows:
            if not item["title"]:
                continue
            matched = community_reports_df[
                community_reports_df["title"].astype(str) == item["title"]
            ] if "title" in community_reports_df.columns else None
            if matched is None or getattr(matched, "empty", True):
                continue
            found = matched.iloc[0]
            if not item["id"]:
                for id_col in existing_id_columns:
                    value = found.get(id_col, "")
                    if str(value).strip():
                        item["id"] = str(value).strip()
                        break
            if not item["summary"]:
                for s_col in ["summary", "content", "full_content"]:
                    value = found.get(s_col, "")
                    if str(value).strip():
                        item["summary"] = str(value).strip()
                        break

    if not rows:
        # 마지막 fallback: assembled_context의 Reports 섹션에서 id만 추출
        ids = _extract_selected_community_ids_from_context(assembled_context)
        rows = [{"id": rid, "title": "", "summary": ""} for rid in ids]

    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in rows:
        key = item["id"] or item["title"]
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _extract_selected_community_ids_from_context(assembled_context: str) -> list[str]:
    if not assembled_context.strip():
        return []
    lines = assembled_context.splitlines()
    in_reports = False
    header_seen = False
    ids: list[str] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if text.startswith("-----") and text.endswith("-----"):
            in_reports = text.strip("-").strip().lower() == "reports"
            header_seen = False
            continue
        if not in_reports or "|" not in text:
            continue
        parts = [p.strip() for p in text.split("|")]
        if not header_seen:
            header_seen = True
            continue
        if parts and parts[0]:
            ids.append(parts[0])
    return list(dict.fromkeys(ids))


def _build_selected_community_context(
    *,
    selected_community_ids: list[Any],
    community_reports_df: Any,
) -> list[str]:
    if not selected_community_ids or community_reports_df is None:
        return []

    id_columns = ["community", "community_id", "id", "short_id", "human_readable_id"]
    summary_columns = ["summary", "full_content", "content"]
    existing_id_columns = [c for c in id_columns if c in getattr(community_reports_df, "columns", [])]
    existing_summary_columns = [
        c for c in summary_columns if c in getattr(community_reports_df, "columns", [])
    ]
    if not existing_id_columns or not existing_summary_columns:
        return []

    summary_col = existing_summary_columns[0]
    contexts: list[str] = []
    for community_id in selected_community_ids:
        matched_row = None
        for id_col in existing_id_columns:
            matched = community_reports_df[
                community_reports_df[id_col].astype(str) == str(community_id)
            ]
            if not matched.empty:
                matched_row = matched.iloc[0]
                break
        if matched_row is None:
            contexts.append(f"[community_id={community_id}]")
            continue
        summary_value = str(matched_row.get(summary_col, "")).strip()
        title_value = str(matched_row.get("title", "")).strip()
        if title_value and summary_value:
            contexts.append(f"[community_id={community_id}] {title_value}: {summary_value}")
        elif summary_value:
            contexts.append(f"[community_id={community_id}] {summary_value}")
        else:
            contexts.append(f"[community_id={community_id}]")
    return contexts


def _clean_text(value: str) -> str:
    cleaned = _DATA_CITATION_RE.sub("", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _strip_explanations(value: str) -> str:
    chunks = [chunk.strip() for chunk in value.split("||")]
    summaries = []
    for chunk in chunks:
        if not chunk:
            continue
        if ":" in chunk:
            summaries.append(chunk.split(":", 1)[0].strip())
        else:
            summaries.append(chunk)
    return " | ".join(summaries[:2]).strip()


def _to_readable_assembled_context(raw_text: str, *, slim: bool = True) -> str:
    if not raw_text.strip():
        return ""

    lines = raw_text.splitlines()
    out_lines: list[str] = []
    current_section = ""
    header: list[str] | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("-----") and line.endswith("-----"):
            current_section = line.strip("-").strip()
            header = None
            out_lines.append(f"[{current_section}]")
            continue
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if header is None:
                header = parts
                continue
            if header and len(parts) == len(header):
                row = {k: v for k, v in zip(header, parts, strict=True)}
                filtered_items = [
                    (k, _clean_text(v))
                    for k, v in row.items()
                    if k.lower() not in _DROP_COLUMNS and _clean_text(v)
                ]
                if not filtered_items:
                    continue
                if slim:
                    filtered_items = [
                        (k, _strip_explanations(v) if k.lower() in {"timeline_events", "superseded_facts"} else v)
                        for k, v in filtered_items
                        if k.lower() in _HUMAN_KEEP_FIELDS
                    ]
                    filtered_items = [
                        (k, v) for k, v in filtered_items if v
                    ]
                    if not filtered_items:
                        continue
                summary = next(
                    (v for k, v in filtered_items if k.lower() in {"summary", "content"}),
                    None,
                )
                if summary:
                    extras = [
                        f"{k}: {v}"
                        for k, v in filtered_items
                        if k.lower() not in {"summary", "content"}
                    ]
                    if extras:
                        out_lines.append(f"- {summary} ({'; '.join(extras)})")
                    else:
                        out_lines.append(f"- {summary}")
                else:
                    out_lines.append(
                        "- " + "; ".join(f"{k}: {v}" for k, v in filtered_items)
                    )
                continue
        cleaned_line = _clean_text(line)
        if cleaned_line:
            out_lines.append(cleaned_line)

    return "\n".join(out_lines).strip()


async def _run_single_query(
    *,
    repo_root: Path,
    output_dir: Path,
    question: str,
    community_policy: str,
    community_level: int,
    response_type: str,
    condition_id: str,
    max_tokens: int | None,
) -> dict[str, Any]:
    cli_overrides = {
        "output_storage": {"base_dir": str(output_dir)},
        "vector_store": {"db_uri": str(output_dir / "lancedb")},
    }
    config = load_config(root_dir=repo_root, cli_overrides=cli_overrides)

    _ = (community_policy, condition_id)
    if max_tokens is not None:
        config.local_search.max_context_tokens = max_tokens

    dfs = _resolve_output_files(
        config=config,
        output_list=["communities", "community_reports", "text_units", "relationships", "entities"],
        optional_list=["covariates"],
    )

    _, context_data = await api.local_search(
        config=config,
        entities=dfs["entities"],
        communities=dfs["communities"],
        community_reports=dfs["community_reports"],
        text_units=dfs["text_units"],
        relationships=dfs["relationships"],
        covariates=dfs.get("covariates"),
        community_level=community_level,
        response_type=response_type,
        query=question,
        context_only=True,
        verbose=False,
    )

    assembled_context = str(context_data.get("context_chunks") or "")
    if not assembled_context and isinstance(context_data.get("context_text"), str):
        assembled_context = str(context_data.get("context_text"))
    selected_rows = _extract_selected_communities(
        context_data=context_data,
        community_reports_df=dfs.get("community_reports"),
        assembled_context=assembled_context,
    )
    selected_community_ids = [row["id"] for row in selected_rows if row["id"]]
    selected_community_summaries = [
        (
            f"[community_id={row['id']}] {row['title']}: {row['summary']}"
            if row["id"] and row["title"] and row["summary"]
            else (f"[community_id={row['id']}] {row['summary']}" if row["id"] and row["summary"] else row["summary"])
        )
        for row in selected_rows
        if row["summary"]
    ]
    context_column = _extract_reports_header_column(assembled_context)
    selected_community_current_states = _extract_report_field_values(
        context_data=context_data,
        assembled_context=assembled_context,
        field_name="current_state",
    )
    selected_community_timeline_events = _extract_report_field_values(
        context_data=context_data,
        assembled_context=assembled_context,
        field_name="timeline_events",
    )
    if not selected_community_current_states:
        selected_community_current_states = _extract_field_values_from_community_reports(
            community_reports_df=dfs.get("community_reports"),
            selected_rows=selected_rows,
            field_name="current_state",
        )
    if not selected_community_timeline_events:
        selected_community_timeline_events = _extract_field_values_from_community_reports(
            community_reports_df=dfs.get("community_reports"),
            selected_rows=selected_rows,
            field_name="timeline_events",
        )
    payload: dict[str, Any] = {
        "condition_id": condition_id,
        "community_policy": community_policy,
        "selected_community_ids": selected_community_ids,
        "assembled_context_tokens": len(
            get_tokenizer(encoding_model=config.chunking.encoding_model).encode(assembled_context)
        ),
        "assembled_context": assembled_context,
        "max_context_tokens": config.local_search.max_context_tokens,
        "configured_use_community_summary": bool(
            getattr(config.local_search, "use_community_summary", False)
        ),
        "detected_community_context_column": context_column,
        "use_community_summary_applied": context_column == "summary",
        "selected_community_context": selected_community_summaries,
        "selected_community_summaries": selected_community_summaries,
        "selected_community_current_states": selected_community_current_states,
        "selected_community_timeline_events": selected_community_timeline_events,
    }
    return payload


def _null_row(
    *,
    test_case: str,
    test_id: str,
    question_type: str,
    question_index: int,
    community_policy: str,
    condition_id: str,
    max_tokens: int | None,
) -> dict[str, Any]:
    return {
        "test_case": test_case,
        "test_id": test_id,
        "question_type": question_type,
        "question_index": question_index,
        "question": "",
        "condition_id": condition_id,
        "community_policy": community_policy,
        "selected_community_ids": [],
        "assembled_context_tokens": 0,
        "assembled_context": "",
        "selected_community_context": [],
        "selected_community_summaries": [],
        "selected_community_current_states": [],
        "selected_community_timeline_events": [],
        "max_context_tokens": max_tokens,
        "configured_use_community_summary": None,
        "detected_community_context_column": "unknown",
        "use_community_summary_applied": False,
        "error": "abstention_skip",
    }


def _write_condition_outputs(rows: list[dict[str, Any]], out_dir: Path, max_tokens: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_max{max_tokens}" if max_tokens is not None else ""

    json_path = out_dir / f"results{suffix}.json"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="input_chat output을 순회하며 experimental local context query를 실행하고 condition별 결과를 집계합니다."
    )
    parser.add_argument("--input-chat-root", type=Path, default=Path("input_chat"), help="input_chat 루트")
    parser.add_argument("--test-case", nargs="+", default=None, help="실행할 test_case 목록(입력 순서 유지)")
    parser.add_argument(
        "--test-ids",
        nargs="+",
        default=None,
        help="실행할 test_id 목록(입력 순서 유지, 미지정 시 test_case 내 전체)",
    )
    parser.add_argument(
        "--policies",
        default="default",
        help="지원 정책 목록(쉼표 구분). 현재 지원 값: default",
    )
    parser.add_argument("--run-id", default=None, help="실행 ID (기본: UTC timestamp)")
    parser.add_argument("--community-level", type=int, default=2, help="local search community level")
    parser.add_argument("--response-type", default="Multiple Paragraphs", help="local search response_type")
    parser.add_argument(
        "--max-tokens",
        type=int,
        nargs="+",
        default=None,
        help="experimental_context_max_tokens 목록 (예: --max-tokens 500 1000 2000)",
    )
    parser.add_argument(
        "--raw-assembled-context",
        action="store_true",
        help="assembled_context를 원문 그대로 저장/출력(기본은 summary 중심 slim 포맷)",
    )
    parser.add_argument("--debug", action="store_true", help="질문 단위 디버그 로그 출력")
    parser.add_argument("--retry-count", type=int, default=3, help="질문 실행 실패 시 최대 재시도 횟수")
    parser.add_argument("--retry-sleep-seconds", type=int, default=60, help="실패 시 재시도 대기 시간(초)")
    parser.add_argument(
        "--show-assembled-context",
        action="store_true",
        help="--debug와 함께 사용 시 assembled_context 본문까지 출력",
    )
    parser.add_argument("--dry-run", action="store_true", help="실제 query 실행 없이 대상/조건만 출력")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_chat_root = (REPO_ROOT / args.input_chat_root).resolve()
    targets = _iter_test_targets_with_filter(
        input_chat_root=input_chat_root,
        test_case_filter=args.test_case,
        test_id_filters=args.test_ids,
    )
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    unsupported = [policy for policy in policies if policy not in SUPPORTED_POLICIES]
    if unsupported:
        raise ValueError(
            f"Unsupported policies requested: {unsupported}. Supported: {SUPPORTED_POLICIES}"
        )
    run_id = args.run_id or datetime.now(UTC).strftime("qfs_%Y%m%dT%H%M%SZ")

    run_root = (REPO_ROOT / "qfs_log" / run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] repo_root={REPO_ROOT}")
    print(f"[INFO] targets={len(targets)}")
    print(f"[INFO] run_root={run_root}")
    max_tokens_values: list[int | None] = args.max_tokens if args.max_tokens else [None]

    if args.debug:
        print(f"[DBG] policies={policies}")
        print(f"[DBG] test_case_filter={args.test_case}")
        print(f"[DBG] test_id_filters={args.test_ids}")
        print(f"[DBG] max_tokens_values={max_tokens_values}")

    results_by_condition: dict[tuple[str, int | None], list[dict[str, Any]]] = defaultdict(list)

    for test_case, test_id, test_id_dir in targets:
        probing_path = test_id_dir / "probing_questions" / "probing_questions.json"
        questions_by_type = _load_questions(probing_path)
        output_dir = test_id_dir / "output"
        if args.debug:
            print(f"\n[DBG] target={test_case}/{test_id}")
            print(f"[DBG] probing_path={probing_path}")
            print(f"[DBG] output_dir={output_dir}")

        for policy in policies:
            condition = _condition_key(policy=policy)
            if args.debug:
                print(f"[DBG] condition={condition}")

            for max_tokens in max_tokens_values:
                if args.debug:
                    print(f"[DBG] max_tokens={max_tokens if max_tokens is not None else 'default'}")
                key = (condition, max_tokens)
                for question_type, question_items in questions_by_type.items():
                    if not isinstance(question_items, list):
                        raise ValueError(
                            f"Invalid question list at {probing_path}: key={question_type}"
                        )

                    for question_index, item in enumerate(question_items, start=1):
                            question = item.get("question") if isinstance(item, dict) else None
                            if not question:
                                raise ValueError(
                                    f"Missing 'question' in {probing_path} ({question_type}[{question_index}])"
                                )
                            if args.debug:
                                print(f"  [DBG] start {question_type}[{question_index}]")

                            token_key = f"t{max_tokens}" if max_tokens is not None else "tdefault"
                            condition_id = (
                                f"{run_id}|{test_case}|{test_id}|{policy}"
                                f"|{token_key}|{question_type}|q{question_index:03d}"
                            )

                            if args.dry_run:
                                print(
                                    f"[DRY-RUN] {test_case}/{test_id} {condition} max_tokens={max_tokens} "
                                    f"{question_type}[{question_index}] {question[:80]}"
                                )
                                continue
                            payload: dict[str, Any] | None = None
                            error_message: str | None = None
                            for attempt in range(1, args.retry_count + 1):
                                try:
                                    payload = asyncio.run(
                                        _run_single_query(
                                            repo_root=REPO_ROOT,
                                            output_dir=output_dir,
                                            question=question,
                                            community_policy=policy,
                                            community_level=args.community_level,
                                            response_type=args.response_type,
                                            condition_id=condition_id,
                                            max_tokens=max_tokens,
                                        )
                                    )
                                    break
                                except Exception as exc:  # noqa: BLE001
                                    error_message = str(exc)
                                    print(
                                        f"[WARN] {test_case}/{test_id} {question_type}[{question_index}] "
                                        f"attempt {attempt}/{args.retry_count} failed: {error_message}"
                                    )
                                    if attempt < args.retry_count:
                                        print(f"[INFO] sleep {args.retry_sleep_seconds}s then retry")
                                        time.sleep(args.retry_sleep_seconds)

                            if payload is None:
                                row = _null_row(
                                    test_case=test_case,
                                    test_id=test_id,
                                    question_type=question_type,
                                    question_index=question_index,
                                    community_policy=policy,
                                    condition_id=condition_id,
                                    max_tokens=max_tokens,
                                )
                                row["question"] = question or ""
                                row["error"] = error_message
                                results_by_condition[key].append(row)
                                continue

                            selected_community_ids = payload.get("selected_community_ids") or []
                            assembled_context_tokens: int | str = payload.get("assembled_context_tokens", 0)
                            assembled_context: str = payload.get("assembled_context") or ""
                            selected_community_context: list[str] | str = payload.get("selected_community_context") or []
                            selected_community_summaries: list[str] | str = payload.get("selected_community_summaries") or []
                            selected_community_current_states: list[str] | str = payload.get("selected_community_current_states") or []
                            selected_community_timeline_events: list[str] | str = payload.get("selected_community_timeline_events") or []
                            use_community_summary_applied = payload.get("use_community_summary_applied")
                            if not args.raw_assembled_context:
                                assembled_context = _to_readable_assembled_context(
                                    str(assembled_context),
                                    slim=True,
                                )
                            if args.debug:
                                print(
                                    "  [DBG] done "
                                    f"tokens={assembled_context_tokens} "
                                    f"selected_community_ids={selected_community_ids}"
                                )
                                if args.show_assembled_context:
                                    print("  [DBG] assembled_context:")
                                    print(
                                        assembled_context
                                        if str(assembled_context).strip()
                                        else "  [empty]"
                                    )
                                    if selected_community_summaries:
                                        print("  [DBG] selected community summaries:")
                                        for i, summary in enumerate(selected_community_summaries, start=1):
                                            print(f"    {i}. {summary}")
                                    if selected_community_current_states:
                                        print("  [DBG] selected community current_state:")
                                        for i, value in enumerate(selected_community_current_states, start=1):
                                            print(f"    {i}. {value}")
                                    if selected_community_timeline_events:
                                        print("  [DBG] selected community timeline_events:")
                                        for i, value in enumerate(selected_community_timeline_events, start=1):
                                            print(f"    {i}. {value}")
                                    print(
                                        "  [DBG] use_community_summary_applied="
                                        f"{use_community_summary_applied}"
                                    )

                            row = {
                                "test_case": test_case,
                                "test_id": test_id,
                                "question_type": question_type,
                                "question_index": question_index,
                                "question": question,
                                "condition_id": condition_id,
                                "community_policy": policy,
                                "selected_community_ids": selected_community_ids,
                                "assembled_context_tokens": assembled_context_tokens,
                                "assembled_context": assembled_context,
                                "selected_community_context": selected_community_context,
                                "selected_community_summaries": selected_community_summaries,
                                "selected_community_current_states": selected_community_current_states,
                                "selected_community_timeline_events": selected_community_timeline_events,
                                "max_context_tokens": max_tokens,
                                "configured_use_community_summary": payload.get(
                                    "configured_use_community_summary"
                                ),
                                "detected_community_context_column": payload.get(
                                    "detected_community_context_column"
                                ),
                                "use_community_summary_applied": payload.get(
                                    "use_community_summary_applied"
                                ),
                                "error": error_message,
                            }
                            results_by_condition[key].append(row)

    if args.dry_run:
        print("[DONE] dry-run only")
        return 0

    for (condition, max_tokens), rows in results_by_condition.items():
        out_dir = run_root / condition
        _write_condition_outputs(rows=rows, out_dir=out_dir, max_tokens=max_tokens)
        suffix = f"_max{max_tokens}" if max_tokens is not None else ""
        print(f"[DONE] {condition} -> {out_dir} (results{suffix}.json)")

    print("\nAll query runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
