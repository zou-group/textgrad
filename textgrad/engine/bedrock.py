try:
    from langchain_aws import ChatBedrock
except ImportError:
    raise ImportError("If you'd like to use Bedrock models, please install the `langchain_aws` package by running `pip install langchain-aws`, and instantiate a Bedrock Client.")

import base64
import json
import os
from typing import Any, Dict, List, Union

from botocore.client import BaseClient
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from platformdirs import user_cache_dir
from tenacity import retry, stop_after_attempt, wait_random_exponential
from .base import CachedEngine, EngineLM
from .engine_utils import get_image_type_from_bytes


class ChatBedrockEngine(EngineLM, CachedEngine):
    SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant"

    def __init__(
        self,
        bedrock_client: BaseClient,
        model_string: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        system_prompt: str = SYSTEM_PROMPT,
        is_multimodal: bool = False,
        **kwargs: Any,
    ):
        root = user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_bedrock_{model_string}.db")
        super().__init__(cache_path=cache_path)
        self.bedrock_client = bedrock_client
        self.client = ChatBedrock(
            client=bedrock_client,
            model_id=model_string,
            model_kwargs=kwargs,
        )
        self.kwargs = kwargs
        self.model_string = model_string
        self.system_prompt = system_prompt
        assert isinstance(self.system_prompt, str)
        self.is_multimodal = is_multimodal

    def __call__(self, prompt: Union[str, List[Union[str, bytes]]], **kwargs):
        passed_through_kwargs = self.kwargs.copy()
        passed_through_kwargs.update(kwargs)
        return self.generate(prompt, **passed_through_kwargs)

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content, system_prompt=None, **kwargs):
        if isinstance(content, str):
            return self._generate_from_single_prompt(content, system_prompt=system_prompt, **kwargs)

        elif isinstance(content, list):
            has_multimodal_input = any(isinstance(item, bytes) for item in content)
            if (has_multimodal_input) and (not self.is_multimodal):
                raise NotImplementedError("Multimodal generation is only supported for Claude-3 and beyond.")

            return self._generate_from_multiple_input(content, system_prompt=system_prompt, **kwargs)

    def _generate_from_single_prompt(self, prompt: str, system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        chat_client = self._update_chat_client(temperature=temperature, max_tokens=max_tokens, top_p=top_p)

        messages = [SystemMessage(content=sys_prompt_arg), HumanMessage(content=prompt)]

        response = chat_client.invoke(messages)

        response_text = str(response.content)
        self._save_cache(sys_prompt_arg + prompt, response_text)
        return response_text

    def _format_content(self, content: List[Union[str, bytes]]) -> List[Union[str, Dict[Any, Any]]]:
        formatted_content: List[Union[str, Dict[Any, Any]]] = []
        for item in content:
            if isinstance(item, bytes):
                image_type = get_image_type_from_bytes(item)

                image_media_type = f"image/{image_type}"
                base64_image = base64.b64encode(item).decode("utf-8")
                formatted_content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": base64_image,
                        },
                    }
                )
            elif isinstance(item, str):
                formatted_content.append({"type": "text", "text": item})
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_from_multiple_input(
        self,
        content: List[Union[str, bytes]],
        system_prompt=None,
        temperature=0,
        max_tokens=2000,
        top_p=0.99,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        cache_key = sys_prompt_arg + json.dumps(formatted_content)
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None:
            return cache_or_none

        chat_client = self._update_chat_client(temperature=temperature, max_tokens=max_tokens, top_p=top_p)

        messages = [SystemMessage(content=sys_prompt_arg), HumanMessage(content=formatted_content)]

        response = chat_client.invoke(messages)

        response_text = str(response.content)
        self._save_cache(cache_key, response_text)
        return response_text

    def _update_chat_client(
        self,
        temperature,
        max_tokens,
        top_p,
    ) -> ChatBedrock:
        chat_client = self.client

        if any(
            [
                self.kwargs.get("temperature", -1) != temperature,
                self.kwargs.get("max_tokens", -1) != max_tokens,
                self.kwargs.get("top_p", -1) != top_p,
            ]
        ):
            updated_kwargs = self.kwargs.copy()
            updated_kwargs["temperature"] = temperature
            updated_kwargs["max_tokens"] = max_tokens
            updated_kwargs["top_p"] = top_p

            chat_client = ChatBedrock(
                client=self.bedrock_client,
                model_id=self.model_string,
                model_kwargs=updated_kwargs,
            )
        return chat_client
