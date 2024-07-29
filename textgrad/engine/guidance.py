try:
    import guidance
except ImportError:
    raise ImportError("Please install the guidance package by running `pip install guidance`.")

import os
import platformdirs

from .base import EngineLM, CachedEngine

class GuidanceEngine(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="meta-llama/Meta-Llama-3-8B-Instruct",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        device="cuda"):
        """
        :param model_string: The model identifier for guidance.models
        :param system_prompt: The system prompt to use
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_guidance_{model_string.replace('/', '_')}.db")
        super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        self.client = guidance.models.Transformers(model_string, device_map={"": device})
        self.model_string = model_string

    def generate(
        self, prompt, system_prompt=None, temperature=0, max_tokens=2000, **kwargs,
    ):
        """
        Generate a response without structured output.
        """
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        lm = self.client
        with guidance.system():
            lm += sys_prompt_arg
        with guidance.user():
            lm += prompt
        with guidance.assistant():
            lm += guidance.gen(name="response", max_tokens=max_tokens, temperature=temperature)

        response_text = lm["response"]

        self._save_cache(sys_prompt_arg + prompt, response_text)
        self.client.reset()
        return response_text

    def generate_structured(self, guidance_structure, **kwargs):
        """
        Generate a response using a provided guidance structure.
        
        :param guidance_structure: A guidance-decorated function defining the structure
        :param kwargs: Additional keyword arguments to pass to the guidance structure
        """
        # TODO: Check if the provided function is decorated with guidance.
        self.client += guidance_structure(**kwargs)
        output_variables = self.client._variables
        self.client.reset()
        return output_variables

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)