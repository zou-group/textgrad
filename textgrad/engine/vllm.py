try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("If you'd like to use hugging face models, please install the vllm package by running `pip install vllm`")



from .base import EngineLM, CachedEngine


class ChatVllm(EngineLM, CachedEngine):
    SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="microsoft/Phi-3-mini-4k-instruct",
        system_prompt=SYSTEM_PROMPT,**kwargs
    ):
                
        self.model_string = model_string
        self.model = LLM(model_string,**kwargs)
        self.tokenizer = AutoTokenizer(model_string)
        self.system_prompt = system_prompt
        assert isinstance(self.system_prompt, str)

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def generate(
        self, prompt, system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99
    ):
        sampling_parameters = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none
        messages = [
                {'role':'system',
                'content':sys_prompt_arg
                },
                {'role':'user',
                'content':prompt
                }
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = self.model.generate([prompt],sampling_params=sampling_parameters)[0].outputs[0]
        response = response.text
        self._save_cache(sys_prompt_arg + prompt, response)
        return response