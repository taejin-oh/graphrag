import asyncio
from types import SimpleNamespace

import pandas as pd

from graphrag.api import query as query_api


async def _mock_streaming_search(**kwargs):
    callbacks = kwargs["callbacks"] or []
    for callback in callbacks:
        callback.on_context({"reports": pd.DataFrame([{"id": "1"}])})

    yield "hello "
    yield "world"


def test_global_search_collects_streamed_chunks_and_context(monkeypatch):
    monkeypatch.setattr(query_api, "init_loggers", lambda **kwargs: None)
    monkeypatch.setattr(query_api, "global_search_streaming", _mock_streaming_search)

    response, context_data = asyncio.run(
        query_api.global_search.raw_function(
            config=SimpleNamespace(),
            entities=pd.DataFrame(),
            communities=pd.DataFrame(),
            community_reports=pd.DataFrame(),
            community_level=None,
            dynamic_community_selection=False,
            response_type="multiple paragraphs",
            query="test",
            callbacks=[],
            verbose=False,
        )
    )

    assert response == "hello world"
    assert "reports" in context_data
