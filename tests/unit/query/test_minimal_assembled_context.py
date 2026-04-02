import asyncio

import pandas as pd
import pytest

from graphrag.cli.query import (
    _build_minimal_assembled_payload,
    _print_minimal_assembled_context,
)
from graphrag.query.context_builder.builders import ContextBuilderResult
from graphrag.query.structured_search.local_search.search import LocalSearch


class _DummyTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()


class _DummyContextBuilder:
    def build_context(self, **kwargs) -> ContextBuilderResult:
        _ = kwargs
        return ContextBuilderResult(
            context_chunks="community summary one\ncommunity summary two",
            context_records={"reports": pd.DataFrame([{"id": "10"}, {"id": "20"}])},
        )


class _FailModel:
    tokenizer = _DummyTokenizer()

    async def completion_async(self, **kwargs):
        _ = kwargs
        raise AssertionError("LLM should not be called in context_only mode")


def test_local_search_context_only_skips_llm_and_returns_context():
    search = LocalSearch(
        model=_FailModel(),
        context_builder=_DummyContextBuilder(),
        tokenizer=_DummyTokenizer(),
    )

    result = asyncio.run(search.search(query="q", context_only=True))

    assert result.response == ""
    assert result.llm_calls == 0
    assert result.context_text == "community summary one\ncommunity summary two"
    assert "reports" in result.context_data


def test_build_minimal_payload_extracts_required_fields():
    payload = _build_minimal_assembled_payload(
        context_data={
            "context_chunks": "A B C",
            "reports": pd.DataFrame([{"id": "1"}, {"id": "2"}, {"id": "1"}]),
        },
        tokenizer=_DummyTokenizer(),
    )

    assert payload["assembled_context"] == "A B C"
    assert payload["assembled_context_tokens"] == 3
    assert payload["selected_community_ids"] == ["1", "2"]


def test_print_minimal_payload_outputs_required_fields(capsys: pytest.CaptureFixture[str]):
    _print_minimal_assembled_context(
        {
            "minimal_assembled_context": {
                "selected_community_ids": ["10"],
                "assembled_context_tokens": 5,
                "assembled_context": "body",
            }
        }
    )
    out = capsys.readouterr().out
    assert "selected_community_ids: ['10']" in out
    assert "assembled_context_tokens: 5" in out
    assert "body" in out
