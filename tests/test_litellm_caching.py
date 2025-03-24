import time
import os
import pytest
from unittest.mock import patch, MagicMock
import diskcache as dc

from textgrad.engine_experimental.litellm import LiteLLMEngine

@pytest.fixture
def cache_setup():
    # Create a temporary cache directory
    cache_dir = "/tmp/textgrad_test_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache = dc.Cache(cache_dir)
    
    # Create engine instances
    engine_with_cache = LiteLLMEngine(
        model_string="gpt-3.5-turbo",
        cache=cache
    )
    
    engine_without_cache = LiteLLMEngine(
        model_string="gpt-3.5-turbo",
        cache=False
    )
    
    yield engine_with_cache, engine_without_cache, cache, cache_dir
    
    if os.path.exists(cache_dir):
        cache.close()
        for root, dirs, files in os.walk(cache_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(cache_dir)

def test_caching_enabled(cache_setup):
    engine_with_cache, _, _, _ = cache_setup
    
    with patch.object(LiteLLMEngine, 'lite_llm_generate') as mock_lite_llm_generate:
        mock_lite_llm_generate.return_value = "This is a test response"
        
        result1 = engine_with_cache.generate("Test prompt", temperature=0.7, max_tokens=50, top_p=1.0)
        
        result2 = engine_with_cache.generate("Test prompt", temperature=0.7, max_tokens=50, top_p=1.0)
        
        assert mock_lite_llm_generate.call_count == 1
        
        assert result1 == result2 == "This is a test response"
        
        engine_with_cache.generate("Test prompt", temperature=0.8, max_tokens=50, top_p=1.0)
        assert mock_lite_llm_generate.call_count == 2

def test_caching_disabled(cache_setup):
    _, engine_without_cache, _, _ = cache_setup
    
    with patch.object(LiteLLMEngine, 'lite_llm_generate') as mock_lite_llm_generate:

        mock_lite_llm_generate.return_value = "This is a test response"
        
        r1 = engine_without_cache.generate("Test prompt", temperature=0.7, max_tokens=50, top_p=1.0)
        
        r2 = engine_without_cache.generate("Test prompt", temperature=0.7, max_tokens=50, top_p=1.0)
        
        assert mock_lite_llm_generate.call_count == 2
        assert r1 == r2 == "This is a test response"

def test_cache_persistence(cache_setup):
    _, _, cache, _ = cache_setup
    
    with patch.object(LiteLLMEngine, 'lite_llm_generate') as mock_lite_llm_generate:
        mock_lite_llm_generate.return_value = "Persistent cache test"
        
        engine1 = LiteLLMEngine(model_string="gpt-3.5-turbo", cache=cache)
        engine1.generate("Persistence test", temperature=0.7, max_tokens=50, top_p=1.0)
        
        assert mock_lite_llm_generate.call_count == 1
        
        engine2 = LiteLLMEngine(model_string="gpt-3.5-turbo", cache=cache)
        engine2.generate("Persistence test", temperature=0.7, max_tokens=50, top_p=1.0)
        
        assert mock_lite_llm_generate.call_count == 1 