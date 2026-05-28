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
        use_datasource=True
    )

    assert request_options["api_version"] == (
        settings_module.AZURE_OPENAI_CLASSIC_ON_YOUR_DATA_API_VERSION
    )
    assert request_options["token_parameter"] == "max_tokens"


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
