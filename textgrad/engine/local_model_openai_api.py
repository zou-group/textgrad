import os
import logging
from .openai import ChatOpenAI

logger = logging.getLogger(__name__)


class ChatExternalClient(ChatOpenAI):
    """
    This is the same as engine.openai.ChatOpenAI, but we pass the
    client explicitly to the constructor.
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
    client = None

    def __init__(
        self,
        client,
        model_string,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        **kwargs,
    ):
        """
        :param client:
        :param model_string:
        :param system_prompt:
        """

        if os.getenv("OPENAI_API_KEY") is None:
            logger.warning("OPENAI_API_KEY not set. Setting it from client.")
            os.environ["OPENAI_API_KEY"] = client.api_key

        super().__init__(
            model_string=model_string, system_prompt=system_prompt, **kwargs
        )
        self.client = client
