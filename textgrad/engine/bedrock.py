try:
    import boto3

except ImportError:
    raise ImportError("If you'd like to use Amazon Bedrock models, please install the boto3 package by running `pip install boto3`")

import os
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from .base import EngineLM, CachedEngine


class ChatBedrock(EngineLM, CachedEngine):
    SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
    
    def __init__(
        self,
        model_string="anthropic.claude-3-sonnet-20240229-v1:0",
        system_prompt=SYSTEM_PROMPT,
        **kwargs
    ):

        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_bedrock_{model_string}.db")
        super().__init__(cache_path=cache_path)
        
        self.model_string = model_string
        self.system_prompt = system_prompt
        self.client = boto3.client(service_name='bedrock-runtime')
        assert isinstance(self.system_prompt, str)

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
    
    def generate_conversation(self, model_id="", system_prompts=[], messages=[], temperature=0.5, top_k=200, top_p=0.99, max_tokens=4096):
        """
        Sends messages to a model.
        Args:
            bedrock_client: The Boto3 Bedrock runtime client.
            model_id (str): The model ID to use.
            system_prompts (JSON) : The system prompts for the model to use.
            messages (JSON) : The messages to send to the model.

        Returns:
            response (JSON): The conversation that the model generated.

        """

        # Base inference parameters to use.
        inference_config = {"temperature": temperature, "topP": top_p, "maxTokens": max_tokens}
        # Additional inference parameters to use.
        additional_model_fields = {"top_k": top_k}
        
        # Send the message.
        response = self.client.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_fields
        )

        return response

    def generate(
        self, prompt, system_prompt=None, temperature=0, max_tokens=4096, top_p=0.99
    ):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        sys_prompt_args = [{"text": sys_prompt_arg}]
        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        messages = [{
        "role": "user",
        "content": [{"text": prompt}]
        }]      
        
        response = self.generate_conversation(self.model_string, system_prompts=sys_prompt_args, messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)


        response = response.text
        self._save_cache(sys_prompt_arg + prompt, response)
        return response