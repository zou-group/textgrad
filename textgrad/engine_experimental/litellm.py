from litellm import completion
from textgrad.engine_experimental.base import EngineLM, cached
import diskcache as dc
from typing import Union, List
from .engine_utils import open_ai_like_formatting
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

class LiteLLMEngine(EngineLM):
    def lite_llm_generate(self, content, system_prompt=None, **kwargs) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        return completion(model=self.model_string,
                          messages=messages)['choices'][0]['message']['content']

    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(self,
                 model_string: str,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 is_multimodal: bool = True,
                 cache=Union[dc.Cache, bool]):

        super().__init__(
            model_string=model_string,
            system_prompt=system_prompt,
            is_multimodal=is_multimodal,
            cache=cache
        )

    @cached
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def _generate_from_single_prompt(
            self, content: str, system_prompt: str = None, temperature=0, max_tokens=2000, top_p=0.99
    ):

        return self.lite_llm_generate(content, system_prompt)

    @cached
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def _generate_from_multiple_input(
            self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99
    ):
        formatted_content = open_ai_like_formatting(content)

        return self.lite_llm_generate(formatted_content, system_prompt)

    def __call__(self, content, **kwargs):
        return self.generate(content, **kwargs)






