import hashlib
import diskcache as dc
from abc import ABC, abstractmethod
from typing import Union, List
import json

class EngineLM(ABC):
    system_prompt: str = "You are a helpful, creative, and smart assistant."
    model_string: str
    @abstractmethod
    def generate(self, prompt, system_prompt=None, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


class CachedEngine:
    def __init__(self, cache_path):
        super().__init__()
        self.cache_path = cache_path
        self.cache = dc.Cache(cache_path)

    def _hash_prompt(self, prompt: str):
        return hashlib.sha256(f"{prompt}".encode()).hexdigest()

    def _check_cache(self, prompt: str):
        if prompt in self.cache:
            return self.cache[prompt]
        else:
            return None

    def _save_cache(self, prompt: str, response: str):
        self.cache[prompt] = response

    def __getstate__(self):
        # Remove the cache from the state before pickling
        state = self.__dict__.copy()
        del state['cache']
        return state

    def __setstate__(self, state):
        # Restore the cache after unpickling
        self.__dict__.update(state)
        self.cache = dc.Cache(self.cache_path)

import platformdirs
import os

class CachedLLM(CachedEngine, EngineLM):
    def __init__(self, model_string, is_multimodal=False, do_cache=False):
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_openai_{model_string}.db")

        super().__init__(cache_path=cache_path)
        self.model_string = model_string
        self.is_multimodal = is_multimodal
        self.do_cache = do_cache

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    @abstractmethod
    def _generate_from_single_prompt(self, prompt: str, system_prompt: str=None, **kwargs):
        pass

    @abstractmethod
    def _generate_from_multiple_input(self, content: List[Union[str, bytes]], system_prompt: str=None, **kwargs):
        pass

    def single_prompt_generate(self, prompt: str, system_prompt: str=None, **kwargs):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        if self.do_cache:
            cache_or_none = self._check_cache(sys_prompt_arg + prompt)
            if cache_or_none is not None:
                return cache_or_none

        response = self._generate_from_single_prompt(prompt, system_prompt=sys_prompt_arg, **kwargs)

        if self.do_cache:
            self._save_cache(sys_prompt_arg + prompt, response)
        return response

    def multimodal_generate(self, content: List[Union[str, bytes]], system_prompt: str = None, **kwargs):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        if self.do_cache:
            key = "".join([str(k) for k in content])

            cache_key = sys_prompt_arg + key
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        response = self._generate_from_multiple_input(content, system_prompt=sys_prompt_arg, **kwargs)

        if self.do_cache:
            self._save_cache(cache_key, response)

        return response

    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt: str = None, **kwargs):
        if isinstance(content, str):
            return self.single_prompt_generate(content, system_prompt=system_prompt, **kwargs)

        elif isinstance(content, list):
            has_multimodal_input = any(isinstance(item, bytes) for item in content)
            if has_multimodal_input and not self.is_multimodal:
                raise NotImplementedError("Multimodal generation is only supported for Claude-3 and beyond.")

            return self.multimodal_generate(content, system_prompt=system_prompt, **kwargs)

