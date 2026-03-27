#!/usr/bin/env python3
"""Extract structured experimental local-context payloads from query.log."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

MARKER = "[LOCAL_CONTEXT_PAYLOAD]"

FIELDS = [
    "condition_id",
    "community_policy",
    "history_enabled",
    "covariate_enabled",
    "query",
    "selected_community_ids",
    "warnings",
    "assembled_context_tokens",
    "assembled_context",
]


def _normalize_row(payload: dict[str, Any]) -> dict[str, Any]:
    row = {field: payload.get(field) for field in FIELDS}
    for list_field in ("selected_community_ids", "warnings"):
        value = row.get(list_field)
        if value is None:
            row[list_field] = []
        elif not isinstance(value, list):
            row[list_field] = [value]
    return row


def parse_payloads(log_path: Path, condition_prefix: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if MARKER not in raw_line:
            continue
        json_blob = raw_line.split(MARKER, maxsplit=1)[1].strip()
        try:
            payload = json.loads(json_blob)
        except json.JSONDecodeError:
            # tolerate lines where non-json suffix got appended
            match = re.search(r"\{.*\}$", json_blob)
            if not match:
                continue
            payload = json.loads(match.group(0))

        condition_id = str(payload.get("condition_id", ""))
        if condition_prefix and not condition_id.startswith(condition_prefix):
            continue

        rows.append(_normalize_row(payload))

    return rows


def _write_assembled_context_files(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows, start=1):
        condition_id = str(row.get("condition_id", f"cond_{i}"))
        safe_condition = re.sub(r"[^a-zA-Z0-9._-]", "_", condition_id)
        path = out_dir / f"{i:03d}_{safe_condition}.txt"
        path.write_text(str(row.get("assembled_context", "")), encoding="utf-8")


def _rows_for_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["selected_community_ids"] = ",".join(map(str, item["selected_community_ids"]))
        item["warnings"] = " | ".join(map(str, item["warnings"]))
        out.append(item)
    return out


def _print_tsv(rows: list[dict[str, Any]]) -> None:
    table_rows = _rows_for_table(rows)
    print("\t".join(FIELDS))
    for row in table_rows:
        print("\t".join(str(row.get(field, "")) for field in FIELDS))


def _print_csv(rows: list[dict[str, Any]]) -> None:
    writer = csv.DictWriter(
        __import__("sys").stdout,
        fieldnames=FIELDS,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in _rows_for_table(rows):
        writer.writerow(row)


def _print_jsonl(rows: list[dict[str, Any]], pretty: bool) -> None:
    for row in rows:
        if pretty:
            print(json.dumps(row, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(row, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "query.log의 [LOCAL_CONTEXT_PAYLOAD] JSON에서 실험 컨텍스트 필드를 추출합니다."
        )
    )
    parser.add_argument("--log", type=Path, default=Path("logs/query.log"), help="query.log 경로")
    parser.add_argument(
        "--format",
        choices=["tsv", "csv", "jsonl"],
        default="tsv",
        help="출력 포맷",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="--format jsonl일 때 들여쓰기된 출력 사용",
    )
    parser.add_argument(
        "--condition-prefix",
        default=None,
        help="condition_id prefix로 필터링",
    )
    parser.add_argument(
        "--assembled-context-dir",
        type=Path,
        default=None,
        help="assembled_context만 별도 txt 파일로 저장할 디렉터리",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.log.exists():
        parser.error(f"log file not found: {args.log}")

    rows = parse_payloads(log_path=args.log, condition_prefix=args.condition_prefix)

    if args.assembled_context_dir:
        _write_assembled_context_files(rows, args.assembled_context_dir)

    if args.format == "tsv":
        _print_tsv(rows)
    elif args.format == "csv":
        _print_csv(rows)
    else:
        _print_jsonl(rows, args.pretty)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
