try:
    import boto3
    from botocore.config import Config

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
        self.system_prompt_supported = True
        if "anthropic" in model_string:
            self.system_prompt_supported = True
        if "meta" in model_string:
            self.system_prompt_supported = True
        if "cohere" in model_string:
            self.system_prompt_supported = True
        if "mistral" in model_string:
            if "instruct" in model_string:
                self.system_prompt_supported = False
            else:
                self.system_prompt_supported = True
        if "amazon" in model_string:
            self.system_prompt_supported = False
            if "premier" in model_string:
                raise ValueError("amazon-titan-premier not supported yet")
        if "ai21" in model_string:
            self.system_prompt_supported = False
            raise ValueError("ai21 not supported yet")
  
        self.max_tokens = kwargs.get("max_tokens", None)
        self.aws_region = kwargs.get("region", None)

        if boto3._get_default_session().get_credentials() is not None:
            if self.aws_region:
                self.my_config = Config(region_name = self.aws_region)
                self.client = boto3.client(service_name='bedrock-runtime', config=self.my_config)
            else:
                self.client = boto3.client(service_name='bedrock-runtime')
        else:
            access_key_id = os.getenv("AWS_ACCESS_KEY_ID", None)
            secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", None)
            session_token = os.getenv("AWS_SESSION_TOKEN", None)
            if self.aws_region is None:
                self.aws_region = os.getenv("AWS_DEFAULT_REGION", None)
                if self.aws_region is None:
                    raise ValueError("AWS region not specified. Please add it in get_engine parameters or has AWS_DEFAULT_REGION var")
            if access_key_id is None:
                raise ValueError("AWS access key ID cannot be 'None'.")
            if secret_access_key is None:
                raise ValueError("AWS secret access key cannot be 'None'.")
            session = boto3.Session(
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                aws_session_token=session_token
            )
            self.my_config = Config(region_name = self.aws_region)
            self.client = session.client(service_name='bedrock-runtime', config=self.my_config)

        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_bedrock_{model_string}.db")
        super().__init__(cache_path=cache_path)
        
        self.model_string = model_string
        self.system_prompt = system_prompt
        
        assert isinstance(self.system_prompt, str)

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
    
    def generate_conversation(self, model_id="", system_prompts=[], messages=[], temperature=0.5, top_k=200, top_p=0.99, max_tokens=2048):
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
        inference_config = {"temperature": temperature, "topP": top_p, "maxTokens": self.max_tokens if self.max_tokens else max_tokens}
        if("anthropic" in model_id): 
            # Additional inference parameters to use.
            additional_model_fields = {"top_k": top_k}
        else: 
            additional_model_fields = {}
        
        # Send the message.
        if self.system_prompt_supported:
            response = self.client.converse(
                modelId=model_id,
                messages=messages,
                system=system_prompts,
                inferenceConfig=inference_config,
                additionalModelRequestFields=additional_model_fields
            )
        else:
            response = self.client.converse(
                modelId=model_id,
                messages=messages,
                inferenceConfig=inference_config,
                additionalModelRequestFields=additional_model_fields
            )

        return response

    def generate(
        self, prompt, system_prompt=None, temperature=0, max_tokens=2048, top_p=0.99
    ):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        sys_prompt_args = [{"text": sys_prompt_arg}]
        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        if self.system_prompt_supported: 
            messages = [{
            "role": "user",
            "content": [{"text": prompt}]
            }]
        else:
            messages = [
            {
            "role": "user",
            "content": [{"text": sys_prompt_arg + "\n\n" + prompt}]
            }]
        
        response = self.generate_conversation(self.model_string, system_prompts=sys_prompt_args, messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

        response = response["output"]["message"]["content"][0]["text"]
        self._save_cache(sys_prompt_arg + prompt, response)
        return response
