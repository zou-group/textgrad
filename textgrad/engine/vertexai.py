try:
    from google.cloud.aiplatform import initializer
    from vertexai.generative_models import (GenerativeModel, 
                                            GenerationConfig,
                                            Part,
                                            Content)
except ImportError:
    raise ImportError("If you'd like to use Vertex models, please install the google-cloud-aiplatform package by running `pip install google-cloud-aiplatform>=1.38`, and init your environment with vertex.init()")

import os
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from .base import EngineLM, CachedEngine


class ChatVertexAI(EngineLM, CachedEngine):
    SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="vertex-gemini-1.5-flash-001",
        system_prompt=SYSTEM_PROMPT,
    ):

        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_vertexai_{model_string}.db")
        super().__init__(cache_path=cache_path)
    
        if not initializer.global_config._project:
            raise ValueError("Please init with vertex.init() if you'd like to use Vertex models.")
        
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
        
        client = GenerativeModel(self.model_string, system_instruction=sys_prompt_arg)

        messages = Content(role="user",
                           parts=[Part.from_text(prompt)])
        
        generation_config =  GenerationConfig(max_output_tokens=max_tokens,
                                                temperature=temperature,
                                                top_p=top_p,
                                                candidate_count=1)
        
        
        response = client.generate_content(messages, 
                                           generation_config=generation_config)


        response = response.text
        self._save_cache(sys_prompt_arg + prompt, response)
        return response