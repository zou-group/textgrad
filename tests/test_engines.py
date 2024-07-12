import pytest

from textgrad.engine import get_engine

def test_ollama_engine():
    # Declare test constants
    OLLAMA_BASE_URL = 'http://localhost:11434/v1'
    MODEL_STRING = "test-model-string"

    # Initialise the engine
    engine = get_engine("ollama-" + MODEL_STRING)

    assert engine
    assert engine.model_string == MODEL_STRING
    assert engine.base_url == OLLAMA_BASE_URL