import os
from importlib import import_module, reload

import pytest


def load_app_module(dotenv_name: str):
    os.environ["DOTENV_PATH"] = os.path.join(
        os.path.dirname(__file__),
        "dotenv_data",
        dotenv_name,
    )
    os.environ[
        "APPLICATIONINSIGHTS_CONNECTION_STRING"
    ] = "InstrumentationKey=00000000-0000-0000-0000-000000000000"
    settings_module = reload(import_module("backend.settings"))
    app_module = reload(import_module("app"))
    return settings_module, app_module


def test_resolve_chat_request_options_uses_classic_api_for_datasource():
    settings_module, app_module = load_app_module(
        "dotenv_with_azure_search_success"
    )

    request_options = app_module.resolve_chat_request_options(
        use_datasource=app_module.should_use_classic_datasource()
    )

    assert app_module.uses_app_managed_azure_search() is True
    assert app_module.should_use_classic_datasource() is False
    assert request_options["api_version"] == (
        settings_module.MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION
    )
    assert request_options["token_parameter"] == "max_completion_tokens"


def test_resolve_chat_request_options_rejects_reasoning_models_with_datasource():
    _, app_module = load_app_module("dotenv_with_azure_search_success")
    app_module.app_settings.azure_openai.configured_model_name = "gpt-5.4"

    with pytest.raises(ValueError, match="does not support reasoning models"):
        app_module.resolve_chat_request_options(use_datasource=True)


def test_resolve_chat_request_options_uses_preview_api_for_reasoning_model_without_datasource():
    settings_module, app_module = load_app_module("dotenv_no_datasource_1")
    app_module.app_settings.azure_openai.configured_model_name = "gpt-5.4"

    request_options = app_module.resolve_chat_request_options(
        use_datasource=False
    )

    assert request_options["api_version"] == (
        settings_module.MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION
    )
    assert request_options["token_parameter"] == "max_completion_tokens"


def test_prepare_model_args_uses_grounding_prompt_for_app_managed_azure_search():
    _, app_module = load_app_module("dotenv_with_azure_search_success")
    app_module.MS_DEFENDER_ENABLED = False

    model_args = app_module.prepare_model_args(
        request_body={
            "messages": [
                {
                    "role": "user",
                    "content": "Where can I request access?",
                }
            ]
        },
        request_headers={},
        token_parameter="max_completion_tokens",
        api_version=app_module.app_settings.azure_openai.preview_api_version,
        use_classic_datasource=False,
        grounding_prompt="Retrieved sources:\n[doc1] Access policy",
    )

    assert "extra_body" not in model_args
    assert model_args["messages"][0]["role"] == "system"
    assert model_args["messages"][0]["content"] == (
        app_module.app_settings.azure_openai.system_message
    )
    assert model_args["messages"][1]["content"] == (
        "Retrieved sources:\n[doc1] Access policy"
    )
    assert model_args["messages"][2]["content"] == "Where can I request access?"
