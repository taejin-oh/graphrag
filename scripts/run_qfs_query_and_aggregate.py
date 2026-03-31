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
]

_DROP_COLUMNS = {"id", "title", "nid"}
_DATA_CITATION_RE = re.compile(r"\[Data:[^\]]+\]")


def _iter_test_targets(input_chat_root: Path, test_case_filter: str | None) -> list[tuple[str, str, Path]]:
    return _iter_test_targets_with_filter(
        input_chat_root=input_chat_root,
        test_case_filter=test_case_filter,
        test_id_filters=None,
    )


def _iter_test_targets_with_filter(
    *,
    input_chat_root: Path,
    test_case_filter: str | None,
    test_id_filters: list[str] | None,
) -> list[tuple[str, str, Path]]:
    if not input_chat_root.exists():
        raise FileNotFoundError(f"input_chat root not found: {input_chat_root}")

    targets: list[tuple[str, str, Path]] = []
    seen_test_ids: set[str] = set()
    for test_case_dir in sorted(p for p in input_chat_root.iterdir() if p.is_dir()):
        test_case = test_case_dir.name
        if test_case_filter and test_case != test_case_filter:
            continue

        for test_id_dir in sorted(p for p in test_case_dir.iterdir() if p.is_dir()):
            test_id = test_id_dir.name
            if test_id_filters and test_id not in test_id_filters:
                continue
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
            seen_test_ids.add(test_id)

    if not targets:
        raise ValueError("No test targets found under input_chat.")
    if test_id_filters:
        missing = [test_id for test_id in test_id_filters if test_id not in seen_test_ids]
        if missing:
            raise FileNotFoundError(
                f"Requested test_id not found under selected scope: {missing}"
            )
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
    return experimental_context.iloc[0].to_dict()


def _clean_text(value: str) -> str:
    cleaned = _DATA_CITATION_RE.sub("", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _to_readable_assembled_context(raw_text: str) -> str:
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

    return _extract_payload(context_data)


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
    }


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    value = out.get("selected_community_ids")
    if isinstance(value, list):
        out["selected_community_ids"] = json.dumps(value, ensure_ascii=False)
    return out


def _write_condition_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_row(row))

    jsonl_path = out_dir / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    json_path = out_dir / "results.json"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="input_chat output을 순회하며 experimental local context query를 실행하고 condition별 결과를 집계합니다."
    )
    parser.add_argument("--input-chat-root", type=Path, default=Path("input_chat"), help="input_chat 루트")
    parser.add_argument("--test-case", default=None, help="특정 test_case만 실행")
    parser.add_argument(
        "--test-ids",
        nargs="+",
        default=None,
        help="실행할 test_id 목록(미지정 시 test_case 내 전체)",
    )
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICIES),
        help="community policy 목록(쉼표 구분)",
    )
    parser.add_argument("--run-id", default=None, help="실행 ID (기본: UTC timestamp)")
    parser.add_argument("--community-level", type=int, default=2, help="local search community level")
    parser.add_argument("--response-type", default="Multiple Paragraphs", help="local search response_type")
    parser.add_argument("--max-tokens", type=int, default=None, help="experimental_context_max_tokens")
    parser.add_argument(
        "--raw-assembled-context",
        action="store_true",
        help="assembled_context를 원문 그대로 저장/출력",
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
    if args.debug:
        print(f"[DBG] policies={policies}")
        print(f"[DBG] test_case_filter={args.test_case}")
        print(f"[DBG] test_id_filters={args.test_ids}")
        print(f"[DBG] max_tokens={args.max_tokens}")

    results_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)

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
                            results_by_condition[condition].append(row)
                            continue

                        question = item.get("question") if isinstance(item, dict) else None
                        if not question:
                            raise ValueError(
                                f"Missing 'question' in {probing_path} ({question_type}[{question_index}])"
                            )
                        if args.debug:
                            print(f"  [DBG] start {question_type}[{question_index}]")

                        condition_id = (
                            f"{run_id}|{test_case}|{test_id}|{policy}|c{int(covariate_enabled)}"
                            f"|{question_type}|q{question_index:03d}"
                        )

                        if args.dry_run:
                            print(
                                f"[DRY-RUN] {test_case}/{test_id} {condition} "
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
                                max_tokens=args.max_tokens,
                            )
                        )

                        selected_community_ids = []
                        assembled_context_tokens: int | str = NULL
                        assembled_context: str = NULL
                        if payload is not None:
                            selected_community_ids = payload.get("selected_community_ids") or []
                            assembled_context_tokens = payload.get("assembled_context_tokens", NULL)
                            assembled_context = payload.get("assembled_context") or ""
                            if not args.raw_assembled_context:
                                assembled_context = _to_readable_assembled_context(
                                    str(assembled_context)
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
                        }
                        results_by_condition[condition].append(row)

    if args.dry_run:
        print("[DONE] dry-run only")
        return 0

    for condition, rows in results_by_condition.items():
        out_dir = run_root / condition
        _write_condition_outputs(rows=rows, out_dir=out_dir)
        print(f"[DONE] {condition} -> {out_dir}")

    print("\nAll query runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
