#!/usr/bin/env python3
"""Run local-search experimental-context condition matrix from query.txt."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "packages" / "graphrag"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import graphrag.api as api
from graphrag.cli.query import _resolve_output_files
from graphrag.config.load_config import load_config

DEFAULT_POLICIES = ["flat_ranked", "leaf_only", "leaf_then_parent_mix", "pyramid"]


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "settings.yaml").exists():
            return candidate
    raise FileNotFoundError(
        "settings.yaml을 찾지 못했습니다. --root로 GraphRAG 프로젝트 루트를 지정하세요."
    )


def _read_queries(query_file: Path) -> list[str]:
    queries = []
    for raw_line in query_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        queries.append(line)
    if not queries:
        raise ValueError("query.txt에서 실행할 쿼리를 찾지 못했습니다.")
    return queries


async def _run(args: argparse.Namespace) -> int:
    cwd = Path.cwd().resolve()
    root = _find_project_root(args.root.resolve() if args.root else cwd)
    data_dir = (args.data.resolve() if args.data else (root / "output").resolve())

    if not args.query_file.exists():
        raise FileNotFoundError(f"query file not found: {args.query_file}")
    if not data_dir.exists():
        raise FileNotFoundError(f"data directory not found: {data_dir}")

    run_id = args.run_id or datetime.now(UTC).strftime("expctx_%Y%m%dT%H%M%SZ")
    run_dir = (args.results_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    queries = _read_queries(args.query_file)
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]

    cli_overrides = {"output_storage": {"base_dir": str(data_dir)}}
    config = load_config(root_dir=root, cli_overrides=cli_overrides)

    dfs = _resolve_output_files(
        config=config,
        output_list=["communities", "community_reports", "text_units", "relationships", "entities"],
        optional_list=["covariates"],
    )

    matrix = [
        (policy, history_enabled, covariate_enabled)
        for policy in policies
        for history_enabled in (False, True)
        for covariate_enabled in (False, True)
    ]

    run_records = []
    for query_idx, query in enumerate(queries, start=1):
        for condition_idx, (policy, history_enabled, covariate_enabled) in enumerate(matrix, start=1):
            condition_core = f"{policy}|h{int(history_enabled)}|c{int(covariate_enabled)}"
            condition_id = f"{run_id}|q{query_idx:03d}|{condition_core}"

            config.local_search.experimental_context_mode = True
            config.local_search.experimental_community_policy = policy
            config.local_search.experimental_history_enabled = history_enabled
            config.local_search.experimental_covariate_enabled = covariate_enabled
            config.local_search.experimental_context_max_tokens = args.max_tokens
            config.local_search.experimental_condition_id = condition_id
            config.local_search.experimental_log_context_payload = True
            config.local_search.experimental_policy_preserve_mode = args.preserve_mode

            response, _ = await api.local_search(
                config=config,
                entities=dfs["entities"],
                communities=dfs["communities"],
                community_reports=dfs["community_reports"],
                text_units=dfs["text_units"],
                relationships=dfs["relationships"],
                covariates=dfs.get("covariates"),
                community_level=args.community_level,
                response_type=args.response_type,
                query=query,
                verbose=args.verbose,
            )

            response_path = run_dir / f"q{query_idx:03d}_c{condition_idx:02d}.response.txt"
            response_path.write_text(str(response), encoding="utf-8")
            run_records.append(
                {
                    "query_index": query_idx,
                    "query": query,
                    "condition_index": condition_idx,
                    "condition_id": condition_id,
                    "community_policy": policy,
                    "history_enabled": history_enabled,
                    "covariate_enabled": covariate_enabled,
                    "response_path": str(response_path),
                }
            )
            print(
                f"[DONE] q={query_idx}/{len(queries)} cond={condition_idx}/{len(matrix)} {condition_id}"
            )

    summary = {
        "run_id": run_id,
        "root": str(root),
        "data_dir": str(data_dir),
        "query_file": str(args.query_file.resolve()),
        "community_level": args.community_level,
        "response_type": args.response_type,
        "max_tokens": args.max_tokens,
        "preserve_mode": args.preserve_mode,
        "conditions_per_query": len(matrix),
        "query_count": len(queries),
        "records": run_records,
        "log_file": str(root / "logs" / "query.log"),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nRun complete: {run_dir}")
    print(f"조건 수(쿼리당): {len(matrix)}")
    print(f"권장 추출 명령:")
    print(
        "  python scripts/extract_experimental_local_context.py "
        f"--log {root / 'logs' / 'query.log'} --condition-prefix {run_id} --format tsv"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="query.txt의 각 줄에 대해 experimental local-context 16개 조건을 자동 실행합니다."
    )
    parser.add_argument("--root", type=Path, default=None, help="GraphRAG 프로젝트 루트(기본: 자동 탐색)")
    parser.add_argument("--data", type=Path, default=None, help="index output 디렉터리(기본: <root>/output)")
    parser.add_argument("--query-file", type=Path, default=Path("query.txt"), help="질의 목록 파일")
    parser.add_argument("--max-tokens", type=int, default=None, help="experimental_context_max_tokens 값")
    parser.add_argument("--community-level", type=int, default=2, help="community_level")
    parser.add_argument("--response-type", default="Multiple Paragraphs", help="response_type")
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICIES),
        help="커뮤니티 정책 목록(쉼표 구분)",
    )
    parser.add_argument(
        "--preserve-mode",
        default="fallback",
        choices=["fallback", "strict"],
        help="policy preserve mode",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experimental_local_context_runs"),
        help="응답/요약 파일 저장 디렉터리",
    )
    parser.add_argument("--run-id", default=None, help="실행 식별자(기본: UTC timestamp)")
    parser.add_argument("--verbose", action="store_true", help="query verbose logging")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
