try:
    from anthropic import Anthropic
except ImportError:
    raise ImportError("If you'd like to use Anthropic models, please install the anthropic package by running `pip install anthropic`, and add 'ANTHROPIC_API_KEY' to your environment variables.")

import os
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from .base import EngineLM, CachedEngine

class ChatAnthropic(EngineLM, CachedEngine):
    SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="claude-3-opus-20240229",
        system_prompt=SYSTEM_PROMPT,
    ):
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_anthropic_{model_string}.db")
        super().__init__(cache_path=cache_path)
        if os.getenv("ANTHROPIC_API_KEY") is None:
            raise ValueError("Please set the ANTHROPIC_API_KEY environment variable if you'd like to use Anthropic models.")
        
        self.client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
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

        response = self.client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_string,
            system=sys_prompt_arg,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response = response.content[0].text
        self._save_cache(sys_prompt_arg + prompt, response)
        return response
