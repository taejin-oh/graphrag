from __future__ import annotations

import csv
import json
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_qfs_index, run_qfs_query_and_aggregate


def test_iter_test_targets_finds_expected_layout(tmp_path: Path) -> None:
    root = tmp_path / "input_chat"
    test_dir = root / "case_a" / "001"
    test_dir.mkdir(parents=True)
    (test_dir / "case_a_001.json").write_text("{}", encoding="utf-8")

    targets = run_qfs_index._iter_test_targets(root, None)

    assert len(targets) == 1
    assert targets[0][0] == "case_a"
    assert targets[0][1] == "001"
    assert targets[0][2] == test_dir


def test_run_single_index_recreates_logs_and_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_id_dir = tmp_path / "input_chat" / "case_a" / "001"
    logs_dir = test_id_dir / "logs"
    output_dir = test_id_dir / "output"
    logs_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    (logs_dir / "old.log").write_text("old", encoding="utf-8")
    (output_dir / "old.txt").write_text("old", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_load_config(*, root_dir: Path, cli_overrides: dict) -> SimpleNamespace:
        captured["root_dir"] = root_dir
        captured["cli_overrides"] = cli_overrides
        return SimpleNamespace()

    class _Output:
        error = None

    async def fake_build_index(**kwargs):
        captured["build_index_kwargs"] = kwargs
        return [_Output()]

    monkeypatch.setattr(run_qfs_index, "load_config", fake_load_config)
    monkeypatch.setattr(run_qfs_index, "init_loggers", lambda **_: None)
    monkeypatch.setattr(run_qfs_index, "validate_config_names", lambda *_: None)
    monkeypatch.setattr(run_qfs_index.api, "build_index", fake_build_index)

    asyncio.run(
        run_qfs_index._run_single_index(
            repo_root=tmp_path,
            test_case="case_a",
            test_id="001",
            test_id_dir=test_id_dir,
            method=run_qfs_index.IndexingMethod.Standard,
            verbose=False,
            skip_validation=False,
            input_type="json",
            input_file_pattern=r".*\.json$",
            force_clean=True,
        )
    )

    assert logs_dir.exists()
    assert output_dir.exists()
    assert not (logs_dir / "old.log").exists()
    assert not (output_dir / "old.txt").exists()

    overrides = captured["cli_overrides"]
    assert overrides["input"]["type"] == "json"
    assert overrides["input"]["file_pattern"] == r".*\.json$"
    assert overrides["input_storage"]["base_dir"] == str(test_id_dir)
    assert overrides["output_storage"]["base_dir"] == str(output_dir)
    assert overrides["reporting"]["base_dir"] == str(logs_dir)


def test_run_single_index_without_force_clean_preserves_existing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_id_dir = tmp_path / "input_chat" / "case_a" / "001"
    logs_dir = test_id_dir / "logs"
    output_dir = test_id_dir / "output"
    logs_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    (logs_dir / "keep.log").write_text("keep", encoding="utf-8")
    (output_dir / "keep.txt").write_text("keep", encoding="utf-8")

    monkeypatch.setattr(run_qfs_index, "load_config", lambda **_: SimpleNamespace())
    monkeypatch.setattr(run_qfs_index, "init_loggers", lambda **_: None)
    monkeypatch.setattr(run_qfs_index, "validate_config_names", lambda *_: None)

    class _Output:
        error = None

    async def fake_build_index(**kwargs):
        return [_Output()]

    monkeypatch.setattr(run_qfs_index.api, "build_index", fake_build_index)

    asyncio.run(
        run_qfs_index._run_single_index(
            repo_root=tmp_path,
            test_case="case_a",
            test_id="001",
            test_id_dir=test_id_dir,
            method=run_qfs_index.IndexingMethod.Standard,
            verbose=False,
            skip_validation=False,
            input_type="json",
            input_file_pattern=r".*\.json$",
            force_clean=False,
        )
    )

    assert (logs_dir / "keep.log").exists()
    assert (output_dir / "keep.txt").exists()


def test_main_resume_skips_success_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path
    input_root = repo_root / "input_chat" / "case_a" / "001"
    input_root.mkdir(parents=True)
    (input_root / "case_a_001.json").write_text("{}", encoding="utf-8")
    (input_root / "logs").mkdir(parents=True, exist_ok=True)
    (input_root / "logs" / "index_status.json").write_text(
        json.dumps({"status": "success", "attempt": 1}), encoding="utf-8"
    )

    monkeypatch.setattr(run_qfs_index, "REPO_ROOT", repo_root)
    called = {"count": 0}

    async def fake_run_single_index(**kwargs):
        called["count"] += 1

    monkeypatch.setattr(run_qfs_index, "_run_single_index", fake_run_single_index)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_qfs_index.py",
            "--test-case",
            "case_a",
            "--resume",
        ],
    )

    rc = run_qfs_index.main()
    assert rc == 0
    assert called["count"] == 0


def test_run_single_query_sets_experimental_flags_and_extracts_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = {
        "selected_community_ids": ["1", "2"],
        "assembled_context_tokens": 77,
        "assembled_context": "context body",
    }

    class _DummyDF:
        empty = False

        class _ILoc:
            def __getitem__(self, _):
                return SimpleNamespace(to_dict=lambda: payload)

        iloc = _ILoc()

    class _LocalSearch:
        experimental_context_mode = False
        experimental_history_enabled = True
        experimental_community_policy = ""
        experimental_covariate_enabled = False
        experimental_condition_id = ""
        experimental_log_context_payload = False
        experimental_context_max_tokens = None

    dummy_config = SimpleNamespace(local_search=_LocalSearch())

    monkeypatch.setattr(run_qfs_query_and_aggregate, "load_config", lambda **_: dummy_config)
    monkeypatch.setattr(
        run_qfs_query_and_aggregate,
        "_resolve_output_files",
        lambda **_: {
            "communities": object(),
            "community_reports": object(),
            "text_units": object(),
            "relationships": object(),
            "entities": object(),
            "covariates": None,
        },
    )

    async def fake_local_search(**kwargs):
        # history 축 제거 검증
        assert kwargs["config"].local_search.experimental_history_enabled is False
        assert kwargs["config"].local_search.experimental_context_mode is True
        assert kwargs["config"].local_search.experimental_community_policy == "flat_ranked"
        assert kwargs["config"].local_search.experimental_covariate_enabled is True
        assert kwargs["config"].local_search.experimental_condition_id == "cond-1"
        return "response", {"experimental_context": _DummyDF()}

    monkeypatch.setattr(run_qfs_query_and_aggregate.api, "local_search", fake_local_search)

    out = asyncio.run(
        run_qfs_query_and_aggregate._run_single_query(
            repo_root=tmp_path,
            output_dir=tmp_path / "output",
            question="Who is X?",
            community_policy="flat_ranked",
            covariate_enabled=True,
            community_level=2,
            response_type="Multiple Paragraphs",
            condition_id="cond-1",
            max_tokens=123,
        )
    )

    assert out == payload


def test_query_main_generates_condition_outputs_and_abstention_null(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path
    input_chat = repo_root / "input_chat" / "case_a" / "001"
    probing_dir = input_chat / "probing_questions"
    probing_dir.mkdir(parents=True)
    (input_chat / "output").mkdir(parents=True)

    probing = {
        "abstention": [{"question": "skip me"}],
        "fact": [{"question": "Who is Scrooge?"}],
    }
    (probing_dir / "probing_questions.json").write_text(
        json.dumps(probing, ensure_ascii=False), encoding="utf-8"
    )

    monkeypatch.setattr(run_qfs_query_and_aggregate, "REPO_ROOT", repo_root)

    async def fake_run_single_query(**kwargs):
        return {
            "selected_community_ids": ["10", "20"],
            "assembled_context_tokens": 321,
            "assembled_context": f"assembled::{kwargs['community_policy']}::{int(kwargs['covariate_enabled'])}",
        }

    monkeypatch.setattr(run_qfs_query_and_aggregate, "_run_single_query", fake_run_single_query)

    run_id = "run_test"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_qfs_query_and_aggregate.py",
            "--run-id",
            run_id,
            "--policies",
            "flat_ranked,leaf_only",
        ],
    )

    rc = run_qfs_query_and_aggregate.main()
    assert rc == 0

    run_root = repo_root / "qfs_log" / run_id
    conditions = {
        "policy=flat_ranked__covariate=off",
        "policy=flat_ranked__covariate=on",
        "policy=leaf_only__covariate=off",
        "policy=leaf_only__covariate=on",
    }

    for condition in conditions:
        csv_path = run_root / condition / "results.csv"
        jsonl_path = run_root / condition / "results.jsonl"
        assert csv_path.exists()
        assert jsonl_path.exists()

        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

        abstention_row = next(r for r in rows if r["question_type"] == "abstention")
        assert abstention_row["question"] == "NULL"
        assert abstention_row["selected_community_ids"] == "NULL"
        assert abstention_row["assembled_context_tokens"] == "NULL"
        assert abstention_row["assembled_context"] == "NULL"

        fact_row = next(r for r in rows if r["question_type"] == "fact")
        assert fact_row["selected_community_ids"] == '["10", "20"]'
        assert fact_row["assembled_context_tokens"] == "321"
        assert "assembled::" in fact_row["assembled_context"]

        with jsonl_path.open("r", encoding="utf-8") as f:
            jsonl_rows = [json.loads(line) for line in f if line.strip()]
        fact_jsonl = next(r for r in jsonl_rows if r["question_type"] == "fact")
        assert fact_jsonl["selected_community_ids"] == ["10", "20"]
