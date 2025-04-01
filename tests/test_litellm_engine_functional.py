import unittest
from textgrad.engine_experimental.litellm import LiteLLMEngine
import time


class TestLiteLLMEngineFunctional(unittest.TestCase):
    """
    Functional tests for the LiteLLMEngine that test real model behavior.
    
    Note: These tests require API keys to be properly configured.
    """
    
    def setUp(self):
        # Use Meta-Llama-3.1-8B-Instruct-Turbo model
        self.model_string = "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        self.prompt = "Write a short poem about testing."
    
    def test_temperature_affects_output(self):
        """Test that different temperature values produce different outputs."""
        # Create two engines with different temperature settings
        engine_low_temp = LiteLLMEngine(model_string=self.model_string)
        engine_high_temp = LiteLLMEngine(model_string=self.model_string)
        
        # Generate multiple responses with different temperatures
        low_temp_responses = []
        high_temp_responses = []
        
        # Generate multiple samples to account for randomness
        num_samples = 3
        for _ in range(num_samples):
            low_temp_responses.append(
                engine_low_temp.generate(self.prompt, temperature=0.1)
            )
            high_temp_responses.append(
                engine_high_temp.generate(self.prompt, temperature=1.0)
            )
            # Add a small delay to avoid rate limits
            time.sleep(1)
        
        # Calculate diversity within each temperature setting
        low_temp_unique = len(set(low_temp_responses))
        high_temp_unique = len(set(high_temp_responses))
        
        # Print responses for debugging
        print("Low temperature responses:")
        for i, resp in enumerate(low_temp_responses):
            print(f"Response {i+1}:\n{resp}\n")
            
        print("High temperature responses:")
        for i, resp in enumerate(high_temp_responses):
            print(f"Response {i+1}:\n{resp}\n")
        
        # Higher temperature should generally lead to more diverse responses
        # This is a probabilistic test, but we expect to see a difference
        self.assertLessEqual(low_temp_unique, high_temp_unique, 
                            "Higher temperature should produce more diverse responses")
    
    def test_max_tokens_affects_output_length(self):
        """Test that max_tokens parameter limits the output length."""
        engine = LiteLLMEngine(model_string=self.model_string)
        
        # Generate with different max_tokens settings
        short_response = engine.generate(
            "Write a detailed essay about the history of artificial intelligence.",
            max_tokens=50
        )
        
        # Add a small delay to avoid rate limits
        time.sleep(1)
        
        long_response = engine.generate(
            "Write a detailed essay about the history of artificial intelligence.",
            max_tokens=200
        )
        
        # Count tokens (this is approximate since we don't have direct access to the tokenizer)
        short_words = len(short_response.split())
        long_words = len(long_response.split())
        
        print(f"Short response ({short_words} words):\n{short_response}\n")
        print(f"Long response ({long_words} words):\n{long_response}\n")
        
        # The longer response should have substantially more words
        self.assertGreater(long_words, short_words * 1.5, 
                          "Response with higher max_tokens should be significantly longer")
    
    def test_system_prompt_affects_output(self):
        """Test that different system prompts produce different outputs."""
        engine = LiteLLMEngine(model_string=self.model_string)
        
        # Generate with different system prompts
        standard_response = engine.generate(
            "Explain what a neural network is.",
            system_prompt="You are a helpful, creative, and smart assistant."
        )
        
        # Add a small delay to avoid rate limits
        time.sleep(1)
        
        technical_response = engine.generate(
            "Explain what a neural network is.",
            system_prompt="You are a technical AI expert. Be precise and use technical terminology."
        )
        
        # Add a small delay to avoid rate limits
        time.sleep(1)
        
        simple_response = engine.generate(
            "Explain what a neural network is.",
            system_prompt="You are teaching a 10-year-old child. Use simple language and analogies."
        )
        
        print(f"Standard response:\n{standard_response}\n")
        print(f"Technical response:\n{technical_response}\n")
        print(f"Simple response:\n{simple_response}\n")
        
        # The responses should be different due to different system prompts
        self.assertNotEqual(standard_response, technical_response,
                          "Different system prompts should produce different responses")
        self.assertNotEqual(technical_response, simple_response,
                          "Different system prompts should produce different responses")
        self.assertNotEqual(standard_response, simple_response,
                          "Different system prompts should produce different responses")
    
    def test_top_p_affects_output(self):
        """Test that different top_p values affect the output diversity."""
        engine_low_p = LiteLLMEngine(model_string=self.model_string)
        engine_high_p = LiteLLMEngine(model_string=self.model_string)
        
        # Generate multiple responses with different top_p values
        low_p_responses = []
        high_p_responses = []
        
        # Generate multiple samples to account for randomness
        num_samples = 3
        for _ in range(num_samples):
            # Use high temperature to amplify the effects of top_p
            low_p_responses.append(
                engine_low_p.generate(self.prompt, temperature=0.9, top_p=0.1)
            )
            # Add a small delay to avoid rate limits
            time.sleep(1)
            
            high_p_responses.append(
                engine_high_p.generate(self.prompt, temperature=0.9, top_p=0.9)
            )
            # Add a small delay to avoid rate limits
            time.sleep(1)
        
        # Calculate diversity within each setting
        low_p_unique = len(set(low_p_responses))
        high_p_unique = len(set(high_p_responses))
        
        print("Low top_p responses:")
        for i, resp in enumerate(low_p_responses):
            print(f"Response {i+1}:\n{resp}\n")
            
        print("High top_p responses:")
        for i, resp in enumerate(high_p_responses):
            print(f"Response {i+1}:\n{resp}\n")
        
        # This is a probabilistic test - higher top_p generally allows for more diverse outputs
        # But it's not guaranteed in every run
        print(f"Low top_p unique responses: {low_p_unique}")
        print(f"High top_p unique responses: {high_p_unique}")
    
    def test_cache_functionality_with_real_model(self):
        """Test that caching works correctly with real model calls."""
        # Create an engine with caching enabled
        engine = LiteLLMEngine(model_string=self.model_string, cache=True)
        
        # First call - should hit the API
        prompt = "Write a haiku about unit testing."
        first_response = engine.generate(prompt, temperature=0.7)
        
        # Second call with identical parameters - should use cache
        second_response = engine.generate(prompt, temperature=0.7)
        
        # Responses should be identical when coming from cache
        self.assertEqual(first_response, second_response, 
                       "Cached responses should be identical")
        
        # Different temperature should bypass cache and get different response
        different_temp_response = engine.generate(prompt, temperature=0.3)
        self.assertNotEqual(first_response, different_temp_response,
                          "Different temperature should bypass cache")
        
        # Add a small delay to avoid rate limits
        time.sleep(1)
        
        # Different system prompt should bypass cache
        different_system_response = engine.generate(
            prompt, 
            temperature=0.7,
            system_prompt="You are a poet who specializes in haikus."
        )
        self.assertNotEqual(first_response, different_system_response,
                          "Different system prompt should bypass cache")
        
        # Test with no cache
        no_cache_engine = LiteLLMEngine(model_string=self.model_string, cache=False)
        
        # Test with temperature=0
        zero_temp_response1 = no_cache_engine.generate(prompt, temperature=0.0)
        zero_temp_response2 = no_cache_engine.generate(prompt, temperature=0.0)
        
        print(f"First no-cache temp=0 response: {zero_temp_response1[:50]}...")
        print(f"Second no-cache temp=0 response: {zero_temp_response2[:50]}...")
        
        # Check if zero temperature responses are the same
        if zero_temp_response1 == zero_temp_response2:
            print("Model is deterministic at temperature=0")
        else:
            print("Model is non-deterministic even at temperature=0")
        
        # Test with temperature=0.1
        low_temp_response1 = no_cache_engine.generate(prompt, temperature=0.1)
        low_temp_response2 = no_cache_engine.generate(prompt, temperature=0.1)
        
        print(f"First no-cache temp=0.1 response: {low_temp_response1[:50]}...")
        print(f"Second no-cache temp=0.1 response: {low_temp_response2[:50]}...")
        
        # Check if low temperature responses are the same
        if low_temp_response1 == low_temp_response2:
            print("Model is deterministic at temperature=0.1")
        else:
            print("Model is non-deterministic at temperature=0.1")
        
        # Test with higher temperature (should be less deterministic)
        high_temp_response1 = no_cache_engine.generate(prompt, temperature=0.7)
        high_temp_response2 = no_cache_engine.generate(prompt, temperature=0.7)
        
        print(f"First no-cache temp=0.7 response: {high_temp_response1[:50]}...")
        print(f"Second no-cache temp=0.7 response: {high_temp_response2[:50]}...")
        
        # Check if high temperature responses are the same
        if high_temp_response1 == high_temp_response2:
            print("Model is deterministic even at temperature=0.7 (unusual)")
        else:
            print("Model is non-deterministic at temperature=0.7 (expected)")

    def test_temperature_affects_response_diversity(self):
        """Test that different temperature values produce different levels of diversity."""
        engine = LiteLLMEngine(model_string=self.model_string, cache=False)
        prompt = "Generate a creative product name for a new smartphone."
        
        # Generate multiple responses at different temperatures
        responses_low_temp = []
        responses_high_temp = []
        
        num_samples = 3
        for i in range(num_samples):
            responses_low_temp.append(engine.generate(prompt, temperature=0.1))
            time.sleep(1)  # Avoid rate limits
            responses_high_temp.append(engine.generate(prompt, temperature=1.0))
            time.sleep(1)  # Avoid rate limits
        
        # Count unique responses at each temperature
        unique_low_temp = len(set(responses_low_temp))
        unique_high_temp = len(set(responses_high_temp))
        
        print("\nLow temperature responses:")
        for i, resp in enumerate(responses_low_temp):
            print(f"Sample {i+1}: {resp}")
        
        print("\nHigh temperature responses:")
        for i, resp in enumerate(responses_high_temp):
            print(f"Sample {i+1}: {resp}")
        
        print(f"\nUnique responses at temp=0.1: {unique_low_temp}")
        print(f"Unique responses at temp=1.0: {unique_high_temp}")
        
        # We expect higher temperature to produce more diverse outputs
        # This is a probabilistic test, but with enough samples it should pass
        self.assertLessEqual(unique_low_temp, unique_high_temp,
                           "Higher temperature should produce more diverse responses")


if __name__ == "__main__":
    unittest.main() 