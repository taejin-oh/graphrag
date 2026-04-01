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
import csv
import json
import re
import sys
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

DEFAULT_POLICIES = ["flat_ranked", "leaf_only", "leaf_then_parent_mix", "pyramid"]
NULL = "NULL"
RESULT_COLUMNS = [
    "test_case",
    "test_id",
    "question_type",
    "question_index",
    "question",
    "community_policy",
    "selected_community_ids",
    "assembled_context_tokens",
    "assembled_context",
    "selected_community_context",
]

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


def _condition_key(policy: str, covariate_enabled: bool) -> str:
    return f"policy={policy}__covariate={'on' if covariate_enabled else 'off'}"


def _extract_payload(context_data: dict[str, Any]) -> dict[str, Any] | None:
    experimental_context = context_data.get("experimental_context")
    if experimental_context is None or not hasattr(experimental_context, "empty"):
        return None
    if experimental_context.empty:
        return None
    # NOTE:
    # `experimental_context` can contain multiple rows when callbacks emit
    # intermediate contexts in a single query lifecycle. We want the final
    # assembled payload for the current query, so pick the last row.
    return experimental_context.iloc[-1].to_dict()


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
    covariate_enabled: bool,
    community_level: int,
    response_type: str,
    condition_id: str,
    max_tokens: int | None,
) -> dict[str, Any] | None:
    cli_overrides = {"output_storage": {"base_dir": str(output_dir)}}
    config = load_config(root_dir=repo_root, cli_overrides=cli_overrides)

    config.local_search.experimental_context_mode = True
    config.local_search.experimental_history_enabled = False
    config.local_search.experimental_community_policy = community_policy
    config.local_search.experimental_covariate_enabled = covariate_enabled
    config.local_search.experimental_condition_id = condition_id
    config.local_search.experimental_log_context_payload = True
    if max_tokens is not None:
        config.local_search.experimental_context_max_tokens = max_tokens

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
        verbose=False,
    )

    payload = _extract_payload(context_data)
    if payload is None:
        return None
    selected_community_ids = payload.get("selected_community_ids") or []
    if isinstance(selected_community_ids, list):
        payload["selected_community_context"] = _build_selected_community_context(
            selected_community_ids=selected_community_ids,
            community_reports_df=dfs.get("community_reports"),
        )
    else:
        payload["selected_community_context"] = []
    return payload


def _null_row(
    *,
    test_case: str,
    test_id: str,
    question_type: str,
    question_index: int,
    community_policy: str,
) -> dict[str, Any]:
    return {
        "test_case": test_case,
        "test_id": test_id,
        "question_type": question_type,
        "question_index": question_index,
        "question": NULL,
        "community_policy": community_policy,
        "selected_community_ids": NULL,
        "assembled_context_tokens": NULL,
        "assembled_context": NULL,
        "selected_community_context": NULL,
    }


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    value = out.get("selected_community_ids")
    if isinstance(value, list):
        out["selected_community_ids"] = json.dumps(value, ensure_ascii=False)
    context_value = out.get("selected_community_context")
    if isinstance(context_value, list):
        out["selected_community_context"] = json.dumps(context_value, ensure_ascii=False)
    return out


def _write_condition_outputs(rows: list[dict[str, Any]], out_dir: Path, max_tokens: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_max{max_tokens}" if max_tokens is not None else ""

    csv_path = out_dir / f"results{suffix}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_row(row))

    jsonl_path = out_dir / f"results{suffix}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

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
        default=",".join(DEFAULT_POLICIES),
        help="community policy 목록(쉼표 구분)",
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
            for covariate_enabled in (False, True):
                condition = _condition_key(policy=policy, covariate_enabled=covariate_enabled)
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
                            if question_type == "abstention":
                                if args.debug:
                                    print(f"  [DBG] skip abstention {question_type}[{question_index}]")
                                row = _null_row(
                                    test_case=test_case,
                                    test_id=test_id,
                                    question_type=question_type,
                                    question_index=question_index,
                                    community_policy=policy,
                                )
                                results_by_condition[key].append(row)
                                continue

                            question = item.get("question") if isinstance(item, dict) else None
                            if not question:
                                raise ValueError(
                                    f"Missing 'question' in {probing_path} ({question_type}[{question_index}])"
                                )
                            if args.debug:
                                print(f"  [DBG] start {question_type}[{question_index}]")

                            token_key = f"t{max_tokens}" if max_tokens is not None else "tdefault"
                            condition_id = (
                                f"{run_id}|{test_case}|{test_id}|{policy}|c{int(covariate_enabled)}"
                                f"|{token_key}|{question_type}|q{question_index:03d}"
                            )

                            if args.dry_run:
                                print(
                                    f"[DRY-RUN] {test_case}/{test_id} {condition} max_tokens={max_tokens} "
                                    f"{question_type}[{question_index}] {question[:80]}"
                                )
                                continue

                            payload = asyncio.run(
                                _run_single_query(
                                    repo_root=REPO_ROOT,
                                    output_dir=output_dir,
                                    question=question,
                                    community_policy=policy,
                                    covariate_enabled=covariate_enabled,
                                    community_level=args.community_level,
                                    response_type=args.response_type,
                                    condition_id=condition_id,
                                    max_tokens=max_tokens,
                                )
                            )

                            selected_community_ids = []
                            assembled_context_tokens: int | str = NULL
                            assembled_context: str = NULL
                            selected_community_context: list[str] | str = NULL
                            if payload is not None:
                                selected_community_ids = payload.get("selected_community_ids") or []
                                assembled_context_tokens = payload.get("assembled_context_tokens", NULL)
                                assembled_context = payload.get("assembled_context") or ""
                                selected_community_context = payload.get("selected_community_context") or []
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

                            row = {
                                "test_case": test_case,
                                "test_id": test_id,
                                "question_type": question_type,
                                "question_index": question_index,
                                "question": question,
                                "community_policy": policy,
                                "selected_community_ids": selected_community_ids,
                                "assembled_context_tokens": assembled_context_tokens,
                                "assembled_context": assembled_context,
                                "selected_community_context": selected_community_context,
                            }
                            results_by_condition[key].append(row)

    if args.dry_run:
        print("[DONE] dry-run only")
        return 0

    for (condition, max_tokens), rows in results_by_condition.items():
        out_dir = run_root / condition
        _write_condition_outputs(rows=rows, out_dir=out_dir, max_tokens=max_tokens)
        suffix = f"_max{max_tokens}" if max_tokens is not None else ""
        file_names = ", ".join(
            [f"results{suffix}.csv", f"results{suffix}.jsonl", f"results{suffix}.json"]
        )
        print(f"[DONE] {condition} -> {out_dir} ({file_names})")

    print("\nAll query runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
