from functools import wraps
from abc import ABC, abstractmethod
import hashlib
from typing import List, Union
import diskcache as dc
import platformdirs
import os



def cached(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):

        if self.cache is False:
            return func(self, *args, **kwargs)

        # get string representation from args and kwargs
        key = hash(str(args) + str(kwargs))
        key = hashlib.sha256(f"{key}".encode()).hexdigest()

        if key in self.cache:
            return self.cache[key]

        result = func(self, *args, **kwargs)
        self.cache[key] = result
        return result

    return wrapper


class EngineLM(ABC):
    system_prompt: str = "You are a helpful, creative, and smart assistant."
    model_string: str
    is_multimodal: bool
    cache: Union[dc.Cache, bool]

    def __init__(self, model_string: str,
                 system_prompt: str = "You are a helpful, creative, and smart assistant.",
                 is_multimodal: bool = False,
                 cache=Union[dc.Cache, bool]):

        """
        Base class for the engines.

        :param model_string: The model string to use.
        :type model_string: str
        :param system_prompt: The system prompt to use. Defaults to "You are a helpful, creative, and smart assistant."
        :type system_prompt: str
        :param is_multimodal: Whether the model is multimodal. Defaults to False.
        :type is_multimodal: bool
        :param cache: The cache to use. Defaults to True. Note that cache can also be a diskcache.Cache object.
        :type cache: Union[diskcache.Cache, bool]
        """

        root = platformdirs.user_cache_dir("textgrad")
        default_cache_path = os.path.join(root, f"cache_model_{model_string}.db")

        self.model_string = model_string
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal

        # cache resolution
        if isinstance(cache, dc.Cache):
            self.cache = cache
        elif cache is True:
            self.cache = dc.Cache(default_cache_path)
        elif cache is False:
            self.cache = False
        else:
            raise ValueError("Cache argument must be a diskcache.Cache object or a boolean.")

    @abstractmethod
    def _generate_from_multiple_input(self, prompt, system_prompt=None, **kwargs) -> str:
        pass

    @abstractmethod
    def _generate_from_single_prompt(self, prompt, system_prompt=None, **kwargs) -> str:
        pass

    def generate(self, content, system_prompt: Union[str | List[Union[str, bytes]]] = None, **kwargs):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        if isinstance(content, str):
            return self._generate_from_single_prompt(content=content, system_prompt=sys_prompt_arg, **kwargs)

        elif isinstance(content, list):
            has_multimodal_input = any(isinstance(item, bytes) for item in content)
            if has_multimodal_input and not self.is_multimodal:
                raise NotImplementedError("Multimodal generation flag is not set, but multimodal input is provided. "
                                          "Is this model multimodal?")

            return self._generate_from_multiple_input(content=content, system_prompt=sys_prompt_arg, **kwargs)

    def __call__(self, *args, **kwargs):
        pass
