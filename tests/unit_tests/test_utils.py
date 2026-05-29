import json
from types import SimpleNamespace

import pytest
from backend.utils import (
    format_as_ndjson,
    format_non_streaming_response,
    format_tool_response,
    parse_multi_columns,
    should_retry_with_max_tokens,
    should_retry_with_max_completion_tokens,
    with_max_tokens,
    with_max_completion_tokens,
)


@pytest.mark.asyncio
async def test_format_as_ndjson():
    async def dummy_generator():
        yield {"message": "test message\n"}

    async for event in format_as_ndjson(dummy_generator()):
        assert event == '{"message": "test message\\n"}\n'


@pytest.mark.asyncio
async def test_format_as_ndjson_exception():
    async def dummy_generator():
        raise Exception("test exception")
        yield {"message": "test message\n"}

    async for event in format_as_ndjson(dummy_generator()):
        assert event == '{"error": "test exception"}'


def test_parse_multi_columns():
    test_pipes = "col1|col2|col3"
    test_commas = "col1,col2,col3"
    test_single = "col1"
    assert parse_multi_columns(test_pipes) == ["col1", "col2", "col3"]
    assert parse_multi_columns(test_commas) == ["col1", "col2", "col3"]
    assert parse_multi_columns(test_single) == ["col1"]


def test_with_max_completion_tokens_replaces_max_tokens():
    model_args = {"messages": [], "max_tokens": 128, "stream": True}

    compat_args = with_max_completion_tokens(model_args)

    assert compat_args["max_completion_tokens"] == 128
    assert "max_tokens" not in compat_args
    assert model_args["max_tokens"] == 128


def test_with_max_tokens_replaces_max_completion_tokens():
    model_args = {"messages": [], "max_completion_tokens": 128, "stream": True}

    compat_args = with_max_tokens(model_args)

    assert compat_args["max_tokens"] == 128
    assert "max_completion_tokens" not in compat_args
    assert model_args["max_completion_tokens"] == 128


def test_should_retry_with_max_completion_tokens_for_unsupported_max_tokens_error():
    error = "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."

    assert should_retry_with_max_completion_tokens(
        error, {"max_tokens": 128}) is True


def test_should_retry_with_max_tokens_for_unsupported_max_completion_tokens_error():
    error = "Unsupported parameter: 'max_completion_tokens' is not supported with this model. Use 'max_tokens' instead."

    assert should_retry_with_max_tokens(
        error, {"max_completion_tokens": 128}) is True


def test_should_retry_with_max_tokens_for_extra_inputs_validation_error():
    error = "Error code: 400 - {'error': {'requestid': '4aa7265f-8daf-45da-b1b6-2be48b1eea8a', 'code': 400, 'message': 'Validation error at #/max_completion_tokens: Extra inputs are not permitted'}}"

    assert should_retry_with_max_tokens(
        error, {"max_completion_tokens": 128}) is False


def test_should_not_retry_with_max_completion_tokens_for_other_errors():
    error = "The request body is invalid."

    assert should_retry_with_max_completion_tokens(
        error, {"max_tokens": 128}) is False


def test_format_tool_response_serializes_tool_context():
    response = format_tool_response(
        {"citations": [{"id": "1"}], "intent": "question"},
        history_metadata={"conversation_id": "conv-1"},
        apim_request_id="req-1",
        response_id="resp-1",
        model="gpt-5.4",
    )

    assert response["choices"][0]["messages"][0]["role"] == "tool"
    assert json.loads(response["choices"][0]["messages"][0]["content"]) == {
        "citations": [{"id": "1"}],
        "intent": "question",
    }


def test_format_non_streaming_response_prefers_explicit_tool_context():
    completion = SimpleNamespace(
        id="resp-1",
        model="gpt-5.4",
        created=1,
        object="chat.completion",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Grounded answer [doc1].",
                    context={"citations": [{"id": "legacy"}]},
                )
            )
        ],
    )

    response = format_non_streaming_response(
        completion,
        history_metadata={},
        apim_request_id="req-1",
        tool_context={"citations": [{"id": "1"}], "intent": "question"},
    )

    assert json.loads(response["choices"][0]["messages"][0]["content"]) == {
        "citations": [{"id": "1"}],
        "intent": "question",
    }
    assert response["choices"][0]["messages"][1]["content"] == "Grounded answer [doc1]."
