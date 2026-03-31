#!/usr/bin/env python3
"""Total QFS pipeline runner (index -> assembled_context query aggregation).

Behavior:
- Runs per target (test_case/test_id):
  1) index
  2) query+aggregate per policy in requested order
- Sleeps after each index/query unit (default: 600s), except the final successful unit.
- Retries failed targets in additional rounds until all succeed (or max-rounds reached).
- Writes concise yellow progress logs to both stdout and progress log file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
YELLOW = "\033[33m"
RESET = "\033[0m"
DEFAULT_POLICIES = ["pyramid", "flat_ranked", "leaf_only", "leaf_then_parent_mix"]


def _log_progress(progress_path: Path, message: str) -> None:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{now}] {message}"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(f"{YELLOW}{line}{RESET}")


def _iter_targets(
    *,
    input_chat_root: Path,
    test_case_filter: str | None,
    test_ids_filter: list[str] | None,
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
            if test_ids_filter and test_id not in test_ids_filter:
                continue

            input_json = test_id_dir / f"{test_case}_{test_id}.json"
            probing_path = test_id_dir / "probing_questions" / "probing_questions.json"
            if not input_json.exists():
                raise FileNotFoundError(
                    f"index input json not found for {test_case}/{test_id}: {input_json}"
                )
            if not probing_path.exists():
                raise FileNotFoundError(
                    f"probing_questions.json not found for {test_case}/{test_id}: {probing_path}"
                )

            targets.append((test_case, test_id, test_id_dir))
            seen_test_ids.add(test_id)

    if not targets:
        raise ValueError("No test targets found under input_chat.")

    if test_ids_filter:
        missing = [test_id for test_id in test_ids_filter if test_id not in seen_test_ids]
        if missing:
            raise FileNotFoundError(f"Requested test_id not found under selected scope: {missing}")

    return targets


def _run_cmd(cmd: list[str]) -> tuple[int, str]:
    completed = subprocess.run(cmd, capture_output=True, text=True)
    tail = (completed.stderr or completed.stdout or "").strip().splitlines()
    short_tail = tail[-1] if tail else ""
    return completed.returncode, short_tail


def _build_index_cmd(*, test_case: str, test_id: str, input_chat_root: Path) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_qfs_index.py"),
        "--input-chat-root",
        str(input_chat_root),
        "--test-case",
        test_case,
        "--test-ids",
        test_id,
        "--continue-on-error",
    ]


def _build_query_cmd(
    *,
    test_case: str,
    test_id: str,
    input_chat_root: Path,
    policy: str,
    run_id: str,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_qfs_query_and_aggregate.py"),
        "--input-chat-root",
        str(input_chat_root),
        "--test-case",
        test_case,
        "--test-ids",
        test_id,
        "--policies",
        policy,
        "--run-id",
        run_id,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="test_case/test_id 단위로 index->query(policies) 전체 파이프라인을 재시도 루프까지 포함해 실행합니다."
    )
    parser.add_argument("--input-chat-root", type=Path, default=Path("input_chat"), help="input_chat 루트")
    parser.add_argument("--test-case", default=None, help="특정 test_case만 실행")
    parser.add_argument("--test-ids", nargs="+", default=None, help="실행할 test_id 목록")
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICIES),
        help="query policy 순서(쉼표 구분). 기본: pyramid,flat_ranked,leaf_only,leaf_then_parent_mix",
    )
    parser.add_argument("--run-id", default=None, help="run id prefix (기본: UTC timestamp)")
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=600,
        help="각 index/query 단위 완료 직후 대기 시간(초). 기본 600",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="재시도 최대 라운드(0이면 무제한)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_chat_root = (REPO_ROOT / args.input_chat_root).resolve()
    run_id = args.run_id or datetime.now(UTC).strftime("qfs_total_%Y%m%dT%H%M%SZ")
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    if not policies:
        raise ValueError("At least one policy is required.")

    progress_path = REPO_ROOT / "qfs_log" / run_id / "progress_log.txt"
    _log_progress(progress_path, f"START run_id={run_id} targets loading")

    targets = _iter_targets(
        input_chat_root=input_chat_root,
        test_case_filter=args.test_case,
        test_ids_filter=args.test_ids,
    )
    _log_progress(progress_path, f"TARGETS count={len(targets)} policies={policies}")

    pending = [(case, tid) for case, tid, _ in targets]
    round_no = 0

    while pending:
        round_no += 1
        if args.max_rounds > 0 and round_no > args.max_rounds:
            _log_progress(progress_path, f"STOP max_rounds reached ({args.max_rounds}), pending={pending}")
            return 1

        _log_progress(progress_path, f"ROUND {round_no} START pending={len(pending)}")
        failed_in_round: list[tuple[str, str]] = []

        for idx, (test_case, test_id) in enumerate(pending, start=1):
            _log_progress(progress_path, f"[{round_no}:{idx}/{len(pending)}] TARGET {test_case}/{test_id} INDEX start")

            index_cmd = _build_index_cmd(
                test_case=test_case,
                test_id=test_id,
                input_chat_root=args.input_chat_root,
            )
            code, tail = _run_cmd(index_cmd)
            if code != 0:
                _log_progress(
                    progress_path,
                    f"[{round_no}:{idx}/{len(pending)}] TARGET {test_case}/{test_id} INDEX fail rc={code} tail={tail}",
                )
                failed_in_round.append((test_case, test_id))
                if args.sleep_seconds > 0:
                    _log_progress(progress_path, f"SLEEP {args.sleep_seconds}s")
                    time.sleep(args.sleep_seconds)
                continue

            _log_progress(progress_path, f"[{round_no}:{idx}/{len(pending)}] TARGET {test_case}/{test_id} INDEX done")
            if args.sleep_seconds > 0:
                _log_progress(progress_path, f"SLEEP {args.sleep_seconds}s")
                time.sleep(args.sleep_seconds)

            query_failed = False
            for policy in policies:
                query_run_id = f"{run_id}__{test_case}_{test_id}__{policy}"
                _log_progress(
                    progress_path,
                    f"[{round_no}:{idx}/{len(pending)}] TARGET {test_case}/{test_id} QUERY policy={policy} start run_id={query_run_id}",
                )
                query_cmd = _build_query_cmd(
                    test_case=test_case,
                    test_id=test_id,
                    input_chat_root=args.input_chat_root,
                    policy=policy,
                    run_id=query_run_id,
                )
                code, tail = _run_cmd(query_cmd)
                if code != 0:
                    _log_progress(
                        progress_path,
                        f"[{round_no}:{idx}/{len(pending)}] TARGET {test_case}/{test_id} QUERY policy={policy} fail rc={code} tail={tail}",
                    )
                    query_failed = True
                    break
                _log_progress(
                    progress_path,
                    f"[{round_no}:{idx}/{len(pending)}] TARGET {test_case}/{test_id} QUERY policy={policy} done",
                )
                is_final_unit = idx == len(pending) and policy == policies[-1] and not failed_in_round
                if args.sleep_seconds > 0 and not is_final_unit:
                    _log_progress(progress_path, f"SLEEP {args.sleep_seconds}s")
                    time.sleep(args.sleep_seconds)

            if query_failed:
                failed_in_round.append((test_case, test_id))

        if not failed_in_round:
            _log_progress(progress_path, f"ALL DONE round={round_no}")
            return 0

        pending = sorted(set(failed_in_round))
        _log_progress(progress_path, f"ROUND {round_no} END failed={len(pending)} next_retry={pending}")

    _log_progress(progress_path, "ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
