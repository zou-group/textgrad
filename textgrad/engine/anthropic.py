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
import base64
import json
from typing import List, Union
from .base import EngineLM, CachedEngine
from .engine_utils import get_image_type_from_bytes

class ChatAnthropic(EngineLM, CachedEngine):
    SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str="claude-3-opus-20240229",
        system_prompt: str=SYSTEM_PROMPT,
        is_multimodal: bool=False,
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
        self.is_multimodal = is_multimodal

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
    
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt: str=None, **kwargs):
        if isinstance(content, str):
            return self._generate_from_single_prompt(content, system_prompt=system_prompt, **kwargs)
        
        elif isinstance(content, list):
            has_multimodal_input = any(isinstance(item, bytes) for item in content)
            if (has_multimodal_input) and (not self.is_multimodal):
                raise NotImplementedError("Multimodal generation is only supported for Claude-3 and beyond.")
            
            return self._generate_from_multiple_input(content, system_prompt=system_prompt, **kwargs)

    def _generate_from_single_prompt(
        self, prompt: str, system_prompt: str=None, temperature=0, max_tokens=2000, top_p=0.99
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

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                image_type = get_image_type_from_bytes(item)

                image_media_type = f"image/{image_type}"
                base64_image = base64.b64encode(item).decode('utf-8')
                formatted_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": base64_image,
                    },
                })
            elif isinstance(item, str):
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_from_multiple_input(
        self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        cache_key = sys_prompt_arg + json.dumps(formatted_content)
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None:
            return cache_or_none

        response = self.client.messages.create(
            model=self.model_string,
            messages=[
                {"role": "user", "content": formatted_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            system=sys_prompt_arg
        )

        response_text = response.content[0].text
        self._save_cache(cache_key, response_text)
        return response_text
