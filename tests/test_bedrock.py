from unittest.mock import Mock

import pytest
from botocore.client import BaseClient
from langchain_aws import ChatBedrock

from textgrad.engine.bedrock import ChatBedrockEngine


@pytest.fixture
def mock_bedrock_client():
    return Mock(spec=BaseClient)


def test_chat_bedrock_engine_init_custom_values(mock_bedrock_client):
    custom_model_kwargs = {"temperature": 0.7, "max_tokens": 1000}
    custom_model_string = "anthropic.claude-3-haiku-20240307-v1:0"
    custom_system_prompt = "You are the best AI assistant ever."

    engine = ChatBedrockEngine(
        bedrock_client=mock_bedrock_client,
        model_string=custom_model_string,
        system_prompt=custom_system_prompt,
        is_multimodal=True,
        **custom_model_kwargs
    )

    assert isinstance(engine.client, ChatBedrock)
    assert engine.model_string == custom_model_string
    assert engine.system_prompt == custom_system_prompt
    assert engine.is_multimodal is True
    assert engine.kwargs == custom_model_kwargs


def test_chat_bedrock_engine_init_default_values(mock_bedrock_client):
    engine = ChatBedrockEngine(bedrock_client=mock_bedrock_client)

    assert isinstance(engine.client, ChatBedrock)
    assert engine.model_string == "anthropic.claude-3-sonnet-20240229-v1:0"
    assert engine.system_prompt == ChatBedrockEngine.SYSTEM_PROMPT
    assert engine.is_multimodal is False
    assert engine.kwargs == {}


def test_chat_bedrock_engine_invalid_system_prompt(mock_bedrock_client):
    with pytest.raises(AssertionError):
        ChatBedrockEngine(bedrock_client=mock_bedrock_client, system_prompt=123)


def test_chat_bedrock_engine_call(mock_bedrock_client):
    model_kwargs = {"temperature": 0.7, "max_tokens": 1000}
    additional_kwargs = {"temperature": 0.8}

    engine = ChatBedrockEngine(bedrock_client=mock_bedrock_client, **model_kwargs)

    engine.generate = Mock(return_value="Mocked response")

    prompt = "Hello, how are you?"
    response = engine(prompt, **additional_kwargs)
    assert response == "Mocked response"
    engine.generate.assert_called_with(prompt, max_tokens=1000, temperature=0.8)

    response = engine(prompt)
    assert response == "Mocked response"
    engine.generate.assert_called_with(prompt, max_tokens=1000, temperature=0.7)


def test_generate_with_string_input(mock_bedrock_client):
    engine = ChatBedrockEngine(bedrock_client=mock_bedrock_client)
    engine._generate_from_single_prompt = Mock(return_value="Mocked response")

    response = engine.generate("Hello, how are you?")

    assert response == "Mocked response"
    engine._generate_from_single_prompt.assert_called_once_with("Hello, how are you?", system_prompt=None)