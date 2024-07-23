try:
    from groq import Groq
except ImportError:
    raise ImportError("If you'd like to use Groq models, please install the groq package by running `pip install groq`, and add 'GROQ_API_KEY' to your environment variables.")

import os
import json
import base64
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union

from .base import EngineLM, CachedEngine
from .engine_utils import get_image_type_from_bytes
from .openai import ChatOpenAI


class ChatGroq(ChatOpenAI):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant1."

    def __init__(
        self,
        model_string: str="groq-llama3-70b-8192",
        system_prompt: str=DEFAULT_SYSTEM_PROMPT,
        **kwargs):
        """
        :param model_string:
        :param system_prompt:
        :param base_url: Used to support Ollama
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_groq_{model_string}.db")
        CachedEngine.__init__(self, cache_path=cache_path)
        
        if os.getenv("GROQ_API_KEY") is None:
            raise ValueError("Please set the GROQ_API_KEY environment variable if you'd like to use Groq models.")
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        self.model_string = model_string
        self.system_prompt = system_prompt
        assert isinstance(self.system_prompt, str)
        self.is_multimodal = False