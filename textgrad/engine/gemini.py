try:
    import google.generativeai as genai

except ImportError:
    raise ImportError("If you'd like to use Gemini models, please install the google-generativeai package by running `pip install google-generativeai`, and add 'GOOGLE_API_KEY' to your environment variables.")

import os
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from .base import EngineLM, CachedEngine


class ChatGemini(EngineLM, CachedEngine):
    SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="gemini-pro",
        system_prompt=SYSTEM_PROMPT,
    ):

        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_gemini_{model_string}.db")
        super().__init__(cache_path=cache_path)
        if os.getenv("GOOGLE_API_KEY") is None:
            raise ValueError("Please set the GOOGLE_API_KEY environment variable if you'd like to use Gemini models.")
        
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        self.model_string = model_string
        self.system_prompt = system_prompt
        assert isinstance(self.system_prompt, str)

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(
        self, prompt, system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99
    ):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none
        
        client = genai.GenerativeModel(self.model_string,
                                       system_instruction=sys_prompt_arg)
        messages = [{'role':'user', 'parts': [prompt]}]
        generation_config =  genai.types.GenerationConfig(max_output_tokens=max_tokens,
                                                          temperature=temperature,
                                                          top_p=top_p,
                                                          candidate_count=1)
        
        
        response = client.generate_content(messages, 
                                           generation_config=generation_config)


        response = response.text
        self._save_cache(sys_prompt_arg + prompt, response)
        return response