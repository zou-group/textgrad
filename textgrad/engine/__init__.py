from .base import EngineLM, CachedEngine

__ENGINE_NAME_SHORTCUTS__ = {
    "opus": "claude-3-opus-20240229",
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-3-sonnet-20240229",
    "sonnet-3.5": "claude-3-5-sonnet-20240620",
    "together-llama-3-70b": "together-meta-llama/Llama-3-70b-chat-hf",
    "ollama-qwen2:0.5b": "qwen2:0.5b",
}

# Any better way to do this?
__MULTIMODAL_ENGINES__ = ["gpt-4-turbo",
                          "gpt-4o",
                          "claude-3-5-sonnet-20240620",
                          "claude-3-opus-20240229",
                          "claude-3-sonnet-20240229",
                          "claude-3-haiku-20240307",
                          "gpt-4-turbo-2024-04-09",
                          ]

def _check_if_multimodal(engine_name: str):
    return any([name == engine_name for name in __MULTIMODAL_ENGINES__])

def validate_multimodal_engine(engine):
    if not _check_if_multimodal(engine.model_string):
        raise ValueError(
            f"The engine provided is not multimodal. Please provide a multimodal engine, one of the following: {__MULTIMODAL_ENGINES__}")

def get_engine(engine_name: str, **kwargs) -> EngineLM:
    if engine_name in __ENGINE_NAME_SHORTCUTS__:
        engine_name = __ENGINE_NAME_SHORTCUTS__[engine_name]

    if "seed" in kwargs and engine_name not in ["gpt-4", "gpt-3.5"]:
        raise ValueError(f"Seed is currently supported only for OpenAI engines, not {engine_name}")

    if "qwen2" in engine_name:
        from .ollama import ChatOllama
        return ChatOllama(model_string=engine_name, **kwargs)
    elif (("gpt-4" in engine_name) or ("gpt-3.5" in engine_name)):
        from .openai import ChatOpenAI
        return ChatOpenAI(model_string=engine_name, is_multimodal=_check_if_multimodal(engine_name), **kwargs)
    elif "claude" in engine_name:
        from .anthropic import ChatAnthropic
        return ChatAnthropic(model_string=engine_name, is_multimodal=_check_if_multimodal(engine_name), **kwargs)
    elif "gemini" in engine_name:
        from .gemini import ChatGemini
        return ChatGemini(model_string=engine_name, **kwargs)
    elif "together" in engine_name:
        from .together import ChatTogether
        engine_name = engine_name.replace("together-", "")
        return ChatTogether(model_string=engine_name, **kwargs)
    elif engine_name in ["command-r-plus", "command-r", "command", "command-light"]:
        from .cohere import ChatCohere
        return ChatCohere(model_string=engine_name, **kwargs)
    else:
        raise ValueError(f"Engine {engine_name} not supported")
