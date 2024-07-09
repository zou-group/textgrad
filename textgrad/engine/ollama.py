try:
    from ollama import Client
except ImportError:
    raise ImportError("If you'd like to use ollama models, please install ollama by going to https://ollama.com/download .")

import os
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import json
from .base import EngineLM, CachedEngine

class ChatOllama(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="qwen2:0.5b",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        **kwargs):
        """
        :param model_string:
        :param system_prompt:
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_ollama_{model_string}.db")

        super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        
        self.client = Client(
            host="http://localhost:11434",
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
        )

        response = response['message']['content']
        self._save_cache(sys_prompt_arg + prompt, response)
        return response

    def generate_with_messages(self, messages, temperature=0, max_tokens=2000, top_p=0.99):
        prompt = json.dumps(messages)

        cache_or_none = self._check_cache(prompt)
        if cache_or_none is not None:
            return cache_or_none

        response = self.client.chat(
            model=self.model_string,
            messages=messages,
        )

        response = response['message']['content']
        self._save_cache(prompt, response)
        return response

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

