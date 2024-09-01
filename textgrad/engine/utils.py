import hashlib

try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    raise ImportError("If you'd like to use OpenAI models, please install the openai package by running `pip install openai`, and add 'OPENAI_API_KEY' to your environment variables.")

import os
import json
from abc import ABC, abstractmethod
import base64
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union
from functools import wraps
from .engine_utils import get_image_type_from_bytes

def cached(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.use_cache:
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
    use_cache: bool = False
    cache_path: str

    @abstractmethod
    def _generate_from_multiple_input(self, prompt, system_prompt=None, **kwargs) -> str:
        pass

    @abstractmethod
    def _generate_from_single_prompt(self, prompt, system_prompt=None, **kwargs) -> str:
        pass

    # TBF this could be simplified to a single generate method
    def _prepare_generate_from_single_prompt(self, prompt: str, system_prompt: str = None, **kwargs):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        return self._generate_from_single_prompt(prompt, system_prompt=sys_prompt_arg, **kwargs)

    def _prepare_generate_from_multiple_input(self, content: List[Union[str, bytes]], system_prompt=None, **kwargs):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        return self._generate_from_multiple_input(content, system_prompt=sys_prompt_arg, **kwargs)

    def generate(self, content, system_prompt=None, **kwargs):
        if isinstance(content, str):
            return self._prepare_generate_from_single_prompt(content, system_prompt=system_prompt, **kwargs)

        elif isinstance(content, list):
            has_multimodal_input = any(isinstance(item, bytes) for item in content)
            if (has_multimodal_input) and (not self.is_multimodal):
                raise NotImplementedError("Multimodal generation is only supported for Claude-3 and beyond.")

            return self._prepare_generate_from_multiple_input(content, system_prompt=system_prompt, **kwargs)

    def __call__(self, *args, **kwargs):
        pass

class OpenAIEngine(EngineLM):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
            self,
            model_string: str = "gpt-3.5-turbo-0613",
            system_prompt: str = DEFAULT_SYSTEM_PROMPT,
            is_multimodal: bool = False,
            use_cache: bool = False,
            base_url: str = None):
        """
        :param model_string:
        :param system_prompt:
        :param base_url: Used to support Ollama
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_openai_{model_string}.db")

        super().__init__(system_prompt=system_prompt,
                         cache_path=cache_path,
                         use_cache=use_cache,
                         is_multimodal=is_multimodal,
                         model_string=model_string)

        self.base_url = base_url

        if not base_url:
            if os.getenv("OPENAI_API_KEY") is None:
                raise ValueError(
                    "Please set the OPENAI_API_KEY environment variable if you'd like to use OpenAI models.")

            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            raise ValueError("Invalid base URL provided. Please use the default OLLAMA base URL or None.")

    @cached
    def _generate_from_single_prompt(
            self, prompt: str, system_prompt: str = None, temperature=0, max_tokens=2000, top_p=0.99
    ):

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response = response.choices[0].message.content
        return response

    @cached
    def _generate_from_multiple_input(
            self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99
    ):
        formatted_content = self._format_content(content)

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response_text = response.choices[0].message.content
        return response_text

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        """Helper function to format a list of strings and bytes into a list of dictionaries to pass as messages to the API.
        """
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                # For now, bytes are assumed to be images
                image_type = get_image_type_from_bytes(item)
                base64_image = base64.b64encode(item).decode('utf-8')
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{base64_image}"
                    }
                })
            elif isinstance(item, str):
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content


class OpenAICompatibleEngine(OpenAIEngine):
    """
        This is the same as engine.openai.ChatOpenAI, but we pass in an external OpenAI client.
        """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
    client = None

    def __init__(
            self,
            client: OpenAI,
            model_string: str = "gpt-3.5-turbo-0613",
            system_prompt: str = DEFAULT_SYSTEM_PROMPT,
            is_multimodal: bool = False,
            use_cache: bool = False,
            base_url: str = None):
        """
        :param client: an OpenAI client object.
        :param model_string: the model name, used for the cache file name and chat completion requests.
        :param system_prompt: the system prompt to use in chat completions.

        Example usage with lm-studio local server, but any client that follows the OpenAI API will work.

        ```python
        from openai import OpenAI
        from textgrad.engine.local_model_openai_api import ChatExternalClient

        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        engine = ChatExternalClient(client=client, model_string="your-model-name")
        print(engine.generate(max_tokens=40, prompt="What is the meaning of life?"))
        ```

        """

        if os.getenv("OPENAI_API_KEY") is None:
            os.environ["OPENAI_API_KEY"] = client.api_key

        self.client = client

        super.__init__(model_string=model_string,
                       system_prompt=system_prompt,
                       is_multimodal=is_multimodal,
                       use_cache=use_cache,
                       base_url=base_url)