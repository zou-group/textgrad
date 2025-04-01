import unittest
from textgrad.engine_experimental.litellm import LiteLLMEngine
import time

class TestLiteLLMEngineParametersAndCaching(unittest.TestCase):
    """
    Tests to verify parameter handling and caching behavior in LiteLLMEngine.
    """
    
    def setUp(self):
        self.model_string = "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        self.prompt = "Write a short poem about coding."
    
    def test_caching_includes_parameters(self):
        """Test to verify that caching correctly incorporates parameters in the cache key."""
        # Create an engine with caching enabled
        engine = LiteLLMEngine(model_string=self.model_string, cache=True)
        
        # First call with temperature 0.2
        first_response = engine.generate(self.prompt, temperature=0.2)
        print(f"First response (temp=0.2): {first_response[:50]}...")
        
        # Second call with same parameters should return cached result
        second_response = engine.generate(self.prompt, temperature=0.2)
        print(f"Second response (temp=0.2): {second_response[:50]}...")
        
        # These should be identical (using cache)
        self.assertEqual(first_response, second_response, 
                        "Same parameters should return cached response")
        
        # Call with different temperature should bypass cache
        third_response = engine.generate(self.prompt, temperature=1.0)
        print(f"Third response (temp=1.0): {third_response[:50]}...")
        
        # These should be different (different parameters)
        self.assertNotEqual(first_response, third_response, 
                           "Different temperature should bypass cache")
        
        # Let's try with a different parameter (top_p)
        fourth_response = engine.generate(self.prompt, temperature=0.2, top_p=0.5)
        print(f"Fourth response (temp=0.2, top_p=0.5): {fourth_response[:50]}...")
        
        # These should be different (different parameters)
        self.assertNotEqual(first_response, fourth_response, 
                           "Different top_p should bypass cache")
    
    def test_deterministic_behavior_with_same_parameters(self):
        """
        Test if using the same parameters multiple times with low temperature 
        produces consistent results (with and without cache).
        """
        # 1. Test with caching OFF
        no_cache_engine = LiteLLMEngine(model_string=self.model_string, cache=False)
        prompt = "Write a short poem about testing."
        
        # Test with temperature = 0.0
        zero_temp_response1 = no_cache_engine.generate(prompt, temperature=0.0)
        print(f"First no-cache response (temp=0.0): {zero_temp_response1[:50]}...")
        
        zero_temp_response2 = no_cache_engine.generate(prompt, temperature=0.0)
        print(f"Second no-cache response (temp=0.0): {zero_temp_response2[:50]}...")
        
        # At temperature=0, responses should be fairly consistent even without cache
        if zero_temp_response1 == zero_temp_response2:
            print("Model is deterministic at temperature 0.0 (good!)")
        else:
            print("Model is NOT deterministic at temperature 0.0")
        
        # Test with temperature = 0.1
        low_temp_response1 = no_cache_engine.generate(prompt, temperature=0.1)
        print(f"First no-cache response (temp=0.1): {low_temp_response1[:50]}...")
        
        low_temp_response2 = no_cache_engine.generate(prompt, temperature=0.1)
        print(f"Second no-cache response (temp=0.1): {low_temp_response2[:50]}...")
        
        # At temperature=0.1, responses may or may not be consistent
        if low_temp_response1 == low_temp_response2:
            print("Model is deterministic at temperature 0.1")
        else:
            print("Model is NOT deterministic at temperature 0.1")
        
        # 2. Now test with caching ON for comparison
        cache_engine = LiteLLMEngine(model_string=self.model_string, cache=True)
        
        # Test with temperature = 0.0 (cached)
        cached_zero_temp1 = cache_engine.generate(prompt, temperature=0.0)
        cached_zero_temp2 = cache_engine.generate(prompt, temperature=0.0)
        
        # These should be identical due to caching
        self.assertEqual(cached_zero_temp1, cached_zero_temp2,
                        "Same parameters with cache should return identical responses")
        
        # Test with temperature = 0.1 (cached)
        cached_low_temp1 = cache_engine.generate(prompt, temperature=0.1)
        cached_low_temp2 = cache_engine.generate(prompt, temperature=0.1)
        
        # These should be identical due to caching
        self.assertEqual(cached_low_temp1, cached_low_temp2,
                        "Same parameters with cache should return identical responses")
        
        # Verify different temperatures produce different results (even with caching)
        self.assertNotEqual(cached_zero_temp1, cached_low_temp1,
                           "Different temperatures should produce different responses")
    
    def test_cache_key_generation(self):
        """Test the cache key generation logic"""
        # Create two different parameter sets
        params1 = {"temperature": 0.2, "model": "test-model"}
        params2 = {"temperature": 0.8, "model": "test-model"}
        
        # Generate keys as our implementation would
        args = ("test prompt",)
        key1 = hash(str(args) + str(params1))
        key2 = hash(str(args) + str(params2))
        
        # Keys should be different for different parameters
        self.assertNotEqual(key1, key2, 
                           "Cache keys should be different for different parameters")
        
        # Same parameters in different order should generate same key
        params3 = {"model": "test-model", "temperature": 0.2}
        key3 = hash(str(args) + str(params3))
        
        # This might fail because dicts don't guarantee order when stringified
        # Just a note that in real implementations, we should normalize parameter order
        print(f"Key1: {key1}")
        print(f"Key3: {key3}")
        print("Note: If these are different, it's because dict ordering affects the hash")

    def test_different_temperatures_produce_different_outputs(self):
        """
        Test that different temperature values actually affect the model output.
        """
        engine = LiteLLMEngine(model_string=self.model_string, cache=False)
        prompt = "Write a creative story about a robot."
        
        # Generate with several different temperatures
        temps = [0.0, 0.1, 0.5, 1.0]
        responses = []
        
        for temp in temps:
            response = engine.generate(prompt, temperature=temp)
            responses.append(response)
            print(f"\nResponse at temp={temp}:\n{response[:100]}...\n")
            time.sleep(1)  # Avoid rate limits
        
        # Check if at least some temperatures produce different outputs
        unique_responses = len(set(responses))
        print(f"Number of unique responses across {len(temps)} temperatures: {unique_responses}")
        
        # We expect at least 2 unique responses with these different temperatures
        # This is a probabilistic test, but it's very likely to pass with these temp values
        self.assertGreater(unique_responses, 1, 
                          "Different temperatures should produce at least some different responses")


if __name__ == "__main__":
    unittest.main() 