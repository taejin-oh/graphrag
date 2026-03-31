#!/usr/bin/env python3
"""Batch index runner for QFS chat inputs.

Usage examples:
  # 기본 실행 (input_chat 전체 순회)
  python scripts/run_qfs_index.py

  # 특정 test_case만 실행
  python scripts/run_qfs_index.py --test-case case_a

  # 실패 지점부터 이어서 실행
  python scripts/run_qfs_index.py --test-case case_a --resume --continue-on-error

  # dry-run (실제 index 실행 없이 대상만 출력)
  python scripts/run_qfs_index.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "packages" / "graphrag"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import graphrag.api as api
from graphrag.callbacks.console_workflow_callbacks import ConsoleWorkflowCallbacks
from graphrag.config.enums import IndexingMethod
from graphrag.config.load_config import load_config
from graphrag.index.validate_config import validate_config_names
from graphrag.logger.standard_logging import init_loggers


def _recreate_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _status_path(test_id_dir: Path) -> Path:
    return test_id_dir / "logs" / "index_status.json"


def _read_status(test_id_dir: Path) -> dict[str, str] | None:
    path = _status_path(test_id_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_status(
    *,
    test_id_dir: Path,
    status: str,
    test_case: str,
    test_id: str,
    attempt: int,
    error_message: str | None = None,
) -> None:
    path = _status_path(test_id_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat()
    prev = _read_status(test_id_dir) or {}
    started_at = prev.get("started_at", now) if status != "running" else now
    payload: dict[str, str | int | None] = {
        "test_case": test_case,
        "test_id": test_id,
        "status": status,
        "attempt": attempt,
        "started_at": started_at,
        "updated_at": now,
        "ended_at": now if status in {"success", "failed"} else None,
        "error_message": error_message,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_test_targets(input_chat_root: Path, test_case_filter: str | None) -> list[tuple[str, str, Path]]:
    if not input_chat_root.exists():
        raise FileNotFoundError(f"input_chat root not found: {input_chat_root}")

    targets: list[tuple[str, str, Path]] = []
    for test_case_dir in sorted(p for p in input_chat_root.iterdir() if p.is_dir()):
        test_case = test_case_dir.name
        if test_case_filter and test_case != test_case_filter:
            continue

        for test_id_dir in sorted(p for p in test_case_dir.iterdir() if p.is_dir()):
            test_id = test_id_dir.name
            input_json = test_id_dir / f"{test_case}_{test_id}.json"
            if not input_json.exists():
                raise FileNotFoundError(
                    f"index input json not found for {test_case}/{test_id}: {input_json}"
                )
            targets.append((test_case, test_id, test_id_dir))

    if not targets:
        raise ValueError("No test targets found under input_chat.")
    return targets


async def _run_single_index(
    *,
    repo_root: Path,
    test_case: str,
    test_id: str,
    test_id_dir: Path,
    method: IndexingMethod,
    verbose: bool,
    skip_validation: bool,
    input_type: str,
    input_file_pattern: str,
    force_clean: bool,
) -> None:
    logs_dir = test_id_dir / "logs"
    output_dir = test_id_dir / "output"

    if force_clean:
        _recreate_dir(logs_dir)
        _recreate_dir(output_dir)
    else:
        logs_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    cli_overrides = {
        "input": {
            "type": input_type,
            "file_pattern": input_file_pattern,
        },
        "input_storage": {"base_dir": str(test_id_dir)},
        "output_storage": {"base_dir": str(output_dir)},
        "reporting": {"base_dir": str(logs_dir)},
    }
    config = load_config(root_dir=repo_root, cli_overrides=cli_overrides)

    init_loggers(config=config, verbose=verbose)
    if not skip_validation:
        validate_config_names(config)

    outputs = await api.build_index(
        config=config,
        method=method,
        is_update_run=False,
        callbacks=[ConsoleWorkflowCallbacks(verbose=verbose)],
        verbose=verbose,
    )
    has_error = any(output.error is not None for output in outputs)
    if has_error:
        raise RuntimeError(
            f"Index failed for {test_case}/{test_id}. Check logs: {logs_dir}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="input_chat/{test_case}/{test_id}를 순회하며 test_id 단위로 index를 실행합니다."
    )
    parser.add_argument(
        "--input-chat-root",
        type=Path,
        default=Path("input_chat"),
        help="input_chat 루트 경로",
    )
    parser.add_argument(
        "--input-type",
        choices=["text", "csv", "json"],
        default="json",
        help="index 입력 파일 타입(기본: json)",
    )
    parser.add_argument(
        "--input-file-pattern",
        default=r".*\.json$",
        help=r"index 입력 파일 regex(기본: .*\.json$)",
    )
    parser.add_argument("--test-case", default=None, help="특정 test_case만 실행")
    parser.add_argument(
        "--method",
        choices=[m.value for m in IndexingMethod],
        default=IndexingMethod.Standard.value,
        help="index method",
    )
    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="config name preflight validation skip",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="대상만 출력하고 실행하지 않음",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="이전 success 대상은 건너뛰고 실패/미실행만 수행",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="실패해도 다음 test_id를 계속 수행",
    )
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="실행 전 logs/output을 삭제 후 재생성",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = REPO_ROOT
    input_chat_root = (repo_root / args.input_chat_root).resolve()

    targets = _iter_test_targets(input_chat_root=input_chat_root, test_case_filter=args.test_case)

    print(f"[INFO] repo_root={repo_root}")
    print(f"[INFO] input_chat_root={input_chat_root}")
    print(f"[INFO] targets={len(targets)}")

    if args.dry_run:
        for test_case, test_id, test_id_dir in targets:
            status = _read_status(test_id_dir)
            status_label = status["status"] if status else "none"
            print(
                f"[DRY-RUN] {test_case}/{test_id} -> {test_id_dir / 'output'} "
                f"(status={status_label})"
            )
        return 0

    method = IndexingMethod(args.method)
    failures: list[tuple[str, str, str]] = []
    for i, (test_case, test_id, test_id_dir) in enumerate(targets, start=1):
        prev_status = _read_status(test_id_dir)
        if args.resume and prev_status and prev_status.get("status") == "success":
            print(f"\n[{i}/{len(targets)}] skip {test_case}/{test_id} (resume: already success)")
            continue

        attempt = int(prev_status.get("attempt", 0)) + 1 if prev_status else 1
        print(f"\n[{i}/{len(targets)}] indexing {test_case}/{test_id}")
        _write_status(
            test_id_dir=test_id_dir,
            status="running",
            test_case=test_case,
            test_id=test_id,
            attempt=attempt,
        )
        try:
            asyncio.run(
                _run_single_index(
                    repo_root=repo_root,
                    test_case=test_case,
                    test_id=test_id,
                    test_id_dir=test_id_dir,
                    method=method,
                    verbose=args.verbose,
                    skip_validation=args.skip_validation,
                    input_type=args.input_type,
                    input_file_pattern=args.input_file_pattern,
                    force_clean=args.force_clean,
                )
            )
            _write_status(
                test_id_dir=test_id_dir,
                status="success",
                test_case=test_case,
                test_id=test_id,
                attempt=attempt,
            )
            print(f"[DONE] {test_case}/{test_id}")
        except Exception as exc:
            _write_status(
                test_id_dir=test_id_dir,
                status="failed",
                test_case=test_case,
                test_id=test_id,
                attempt=attempt,
                error_message=str(exc),
            )
            failures.append((test_case, test_id, str(exc)))
            print(f"[ERROR] {test_case}/{test_id}: {exc}")
            if not args.continue_on_error:
                break

    if failures:
        print("\nIndex run finished with failures:")
        for test_case, test_id, message in failures:
            print(f" - {test_case}/{test_id}: {message}")
        return 1

    print("\nAll index runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
