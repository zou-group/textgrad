import os
import logging
from openai import OpenAI
from .openai import ChatOpenAI

logger = logging.getLogger(__name__)


class ChatExternalClient(ChatOpenAI):
    """
    This is the same as engine.openai.ChatOpenAI, but we pass in an external OpenAI client.
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
    client = None

    def __init__(
        self,
        client: OpenAI,
        model_string: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        **kwargs,
    ):
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
            logger.warning("OPENAI_API_KEY not set. Setting it from client.")
            os.environ["OPENAI_API_KEY"] = client.api_key

        super().__init__(
            model_string=model_string, system_prompt=system_prompt, **kwargs
        )
        self.client = client
