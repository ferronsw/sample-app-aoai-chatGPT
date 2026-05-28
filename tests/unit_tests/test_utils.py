import pytest
from backend.utils import (
    format_as_ndjson,
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
        error, {"max_completion_tokens": 128}) is True


def test_should_not_retry_with_max_completion_tokens_for_other_errors():
    error = "The request body is invalid."

    assert should_retry_with_max_completion_tokens(
        error, {"max_tokens": 128}) is False
