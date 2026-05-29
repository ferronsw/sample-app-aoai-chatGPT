import os
from importlib import import_module, reload
from types import SimpleNamespace

import pytest


def load_app_settings(dotenv_name: str):
    os.environ["DOTENV_PATH"] = os.path.join(
        os.path.dirname(__file__),
        "dotenv_data",
        dotenv_name,
    )
    settings_module = reload(import_module("backend.settings"))
    return getattr(settings_module, "app_settings")


class DummyAsyncSearchResults:
    def __init__(self, items):
        self._items = items
        self._index = 0

    def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration

        item = self._items[self._index]
        self._index += 1
        return item


@pytest.mark.asyncio
async def test_retrieve_and_format_sources_uses_vector_hybrid_and_shapes_citations(monkeypatch):
    app_settings = load_app_settings("dotenv_with_azure_search_success")
    retrieval_module = reload(import_module("backend.azure_search_retrieval"))
    captured_calls = {}

    class DummySearchClient:
        def __init__(self, **kwargs):
            captured_calls["client_init"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def search(self, *args, **kwargs):
            captured_calls["search"] = {"args": args, "kwargs": kwargs}
            return DummyAsyncSearchResults(
                [
                    {
                        "content1": "Contoso access requests are approved in the portal.",
                        "content2": "Use the employee support workflow.",
                        "title": "Access policy",
                        "filepath": "/docs/access-policy.md",
                        "url": "https://contoso.example/access-policy",
                        "@search.score": 3.14,
                    }
                ]
            )

    async def fake_create_embedding(query_text, _app_settings):
        captured_calls["embedding_query"] = query_text
        return [0.1, 0.2]

    monkeypatch.setattr(retrieval_module, "SearchClient", DummySearchClient)
    monkeypatch.setattr(
        retrieval_module, "_create_embedding", fake_create_embedding)

    result = await retrieval_module.retrieve_and_format_sources(
        [{"role": "user", "content": "How do I request access?"}],
        SimpleNamespace(headers={}),
        app_settings,
    )

    assert result.query_type == "vector_simple_hybrid"
    assert captured_calls["embedding_query"] == "How do I request access?"
    assert captured_calls["search"]["kwargs"]["vector_fields"] == "vector1"
    assert captured_calls["search"]["kwargs"]["top"] == 5
    assert result.citations[0]["title"] == "Access policy"
    assert result.citations[0]["filepath"] == "/docs/access-policy.md"
    assert "[doc1]" in result.prompt
    assert result.tool_message()["intent"] == "How do I request access?"


@pytest.mark.asyncio
async def test_retrieve_and_format_sources_returns_fallback_when_no_results(monkeypatch):
    app_settings = load_app_settings("dotenv_with_azure_search_success")
    retrieval_module = reload(import_module("backend.azure_search_retrieval"))

    class DummySearchClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def search(self, *args, **kwargs):
            return DummyAsyncSearchResults([])

    async def fake_create_embedding(query_text, _app_settings):
        return [0.1, 0.2]

    monkeypatch.setattr(retrieval_module, "SearchClient", DummySearchClient)
    monkeypatch.setattr(
        retrieval_module, "_create_embedding", fake_create_embedding)

    result = await retrieval_module.retrieve_and_format_sources(
        [{"role": "user", "content": "How do I request access?"}],
        SimpleNamespace(headers={}),
        app_settings,
    )

    assert result.citations == []
    assert result.prompt is None
    assert result.assistant_fallback == retrieval_module.DEFAULT_NO_RESULTS_MESSAGE
