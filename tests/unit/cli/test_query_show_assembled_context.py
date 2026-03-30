# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

from __future__ import annotations

from typer.testing import CliRunner

from graphrag.cli.main import app
from graphrag.cli.query import _print_assembled_context


def test_print_assembled_context_outputs_payload(capsys):
    import pandas as pd

    context_data = {
        "experimental_context": pd.DataFrame(
            [
                {
                    "condition_id": "manual_q001_c01",
                    "assembled_context_tokens": 123,
                    "warnings": ["warn-1"],
                    "assembled_context": "A\nB",
                }
            ]
        )
    }

    _print_assembled_context(context_data)
    out = capsys.readouterr().out

    assert "===== assembled_context =====" in out
    assert "condition_id: manual_q001_c01" in out
    assert "assembled_context_tokens: 123" in out
    assert "warnings: ['warn-1']" in out
    assert "A\nB" in out
    assert "===== /assembled_context =====" in out


def test_print_assembled_context_handles_missing_payload(capsys):
    _print_assembled_context({})
    out = capsys.readouterr().out
    assert "experimental_context payload not found" in out


def test_query_cli_passes_show_assembled_context_flag(monkeypatch, tmp_path):
    runner = CliRunner()
    called: dict[str, object] = {}

    def _fake_run_local_search(**kwargs):
        called.update(kwargs)
        return "ok", {}

    monkeypatch.setattr("graphrag.cli.query.run_local_search", _fake_run_local_search)

    result = runner.invoke(
        app,
        [
            "query",
            "hello",
            "--method",
            "local",
            "--show-assembled-context",
            "--root",
            str(tmp_path),
            "--data",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert called["show_assembled_context"] is True
