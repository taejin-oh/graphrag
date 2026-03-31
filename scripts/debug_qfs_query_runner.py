#!/usr/bin/env python3
"""Debug runner for identifying where run_qfs_query_and_aggregate may stall.

Usage examples:
  python scripts/debug_qfs_query_runner.py --test-case 100K --test-ids 7 --max-tokens 1000
  python scripts/debug_qfs_query_runner.py --test-case 100K --test-ids 7 --per-query-timeout 120 --continue-on-timeout
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_qfs_query_and_aggregate as base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QFS query runner의 멈춤 지점을 단계별로 진단합니다."
    )
    parser.add_argument("--input-chat-root", type=Path, default=Path("input_chat"))
    parser.add_argument("--test-case", required=True, help="대상 test_case")
    parser.add_argument("--test-ids", nargs="+", required=True, help="대상 test_id 목록")
    parser.add_argument(
        "--policies",
        default=",".join(base.DEFAULT_POLICIES),
        help="community policy 목록(쉼표 구분)",
    )
    parser.add_argument("--community-level", type=int, default=2)
    parser.add_argument("--response-type", default="Multiple Paragraphs")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--per-query-timeout",
        type=float,
        default=180.0,
        help="질문 1개당 최대 대기 시간(초)",
    )
    parser.add_argument(
        "--continue-on-timeout",
        action="store_true",
        help="timeout 발생 시 다음 질문으로 계속 진행",
    )
    parser.add_argument(
        "--show-assembled-context",
        action="store_true",
        help="각 질문 완료 시 assembled_context 본문까지 출력",
    )
    return parser


async def _run_with_timeout(timeout_sec: float, **kwargs):
    return await asyncio.wait_for(base._run_single_query(**kwargs), timeout=timeout_sec)


def main() -> int:
    args = build_parser().parse_args()

    input_chat_root = (base.REPO_ROOT / args.input_chat_root).resolve()
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]

    targets = base._iter_test_targets_with_filter(
        input_chat_root=input_chat_root,
        test_case_filter=args.test_case,
        test_id_filters=args.test_ids,
    )

    print(f"[DBG] repo_root={base.REPO_ROOT}")
    print(f"[DBG] input_chat_root={input_chat_root}")
    print(f"[DBG] targets={[(c, t) for c, t, _ in targets]}")

    for test_case, test_id, test_id_dir in targets:
        probing_path = test_id_dir / "probing_questions" / "probing_questions.json"
        questions_by_type = base._load_questions(probing_path)
        output_dir = test_id_dir / "output"

        print(f"\n[DBG] target={test_case}/{test_id}")
        print(f"[DBG] probing_path={probing_path}")
        print(f"[DBG] output_dir={output_dir}")

        for policy in policies:
            for covariate_enabled in (False, True):
                condition = base._condition_key(policy=policy, covariate_enabled=covariate_enabled)
                print(f"[DBG] condition={condition}")

                for question_type, question_items in questions_by_type.items():
                    if not isinstance(question_items, list):
                        raise ValueError(
                            f"Invalid question list at {probing_path}: key={question_type}"
                        )

                    for question_index, item in enumerate(question_items, start=1):
                        if question_type == "abstention":
                            print(f"  [DBG] skip abstention {question_type}[{question_index}]")
                            continue

                        question = item.get("question") if isinstance(item, dict) else None
                        if not question:
                            raise ValueError(
                                f"Missing 'question' in {probing_path} ({question_type}[{question_index}])"
                            )

                        condition_id = (
                            f"debug|{test_case}|{test_id}|{policy}|c{int(covariate_enabled)}"
                            f"|{question_type}|q{question_index:03d}"
                        )
                        print(
                            f"  [DBG] start {question_type}[{question_index}] timeout={args.per_query_timeout}s"
                        )
                        t0 = time.perf_counter()
                        try:
                            payload = asyncio.run(
                                _run_with_timeout(
                                    timeout_sec=args.per_query_timeout,
                                    repo_root=base.REPO_ROOT,
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
                            dt = time.perf_counter() - t0
                            if payload is None:
                                print(f"  [DBG] done in {dt:.2f}s payload=None")
                            else:
                                print(
                                    "  [DBG] done in "
                                    f"{dt:.2f}s tokens={payload.get('assembled_context_tokens')} "
                                    f"selected_community_ids={payload.get('selected_community_ids')}"
                                )
                                if args.show_assembled_context:
                                    assembled_context = str(payload.get("assembled_context") or "")
                                    print("  [DBG] assembled_context:")
                                    print(assembled_context if assembled_context.strip() else "  [empty]")
                        except TimeoutError:
                            dt = time.perf_counter() - t0
                            print(
                                f"  [DBG][TIMEOUT] {question_type}[{question_index}] after {dt:.2f}s"
                            )
                            if not args.continue_on_timeout:
                                return 124

    print("\n[DBG] completed without timeout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
