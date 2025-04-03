try:
    from mistralai.client import MistralClient
except ImportError:
    raise ImportError("If you'd like to use Mistral models, please install the mistralai package by running `pip install mistralai`, and add 'MISTRAL_API_KEY' to your environment variables.")

import os
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from .base import EngineLM, CachedEngine

class ChatMistral(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="open-mixtral-8x7b",
        system_prompt=DEFAULT_SYSTEM_PROMPT):
        """
        :param model_string:
        :param system_prompt:
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_mistral_{model_string}.db")
        super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        if os.getenv("MISTRAL_API_KEY") is None:
            raise ValueError("Please set the MISTRAL_API_KEY environment variable if you'd like to use MistralAI models.")
        
        self.client = MistralClient(
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
        self.model_string = model_string

    def generate(
        self, prompt, system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99
    ):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        response = self.client.chat(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response = response.choices[0].message.content
        self._save_cache(sys_prompt_arg + prompt, response)
        return response

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

