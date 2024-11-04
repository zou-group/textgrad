from abc import ABC, abstractmethod
from typing import List, Union
from collections import defaultdict
from textgrad.variable import Variable
from textgrad import logger
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default
from .optimizer_prompts import construct_tgd_prompt, OPTIMIZER_SYSTEM_PROMPT, GRADIENT_TEMPLATE, GRADIENT_MULTIPART_TEMPLATE
from .optimizer import TextualGradientDescent, Optimizer, get_gradient_and_context_text
try:
    from guidance import models, gen
    import guidance
except ImportError:
    raise ImportError(
        "If you'd like to use guided optimization with guidance, please install the package by running `pip install guidance`."
    )


from textgrad.engine.guidance import GuidanceEngine


class GuidedTextualGradientDescent(TextualGradientDescent):
    def __init__(self, 
                 parameters: List[Variable], 
                 verbose: int=0, 
                 engine: Union[GuidanceEngine, str]=None, 
                 constraints: List[str]=None,
                 new_variable_tags: List[str]=None,
                 optimizer_system_prompt: str=OPTIMIZER_SYSTEM_PROMPT,
                 in_context_examples: List[str]=None,
                 gradient_memory: int=0):
        """GuidedTextualGradientDescent optimizer

        :param engine: the engine to use for updating variables
        :type engine: EngineLM
        :param parameters: the parameters to optimize
        :type parameters: List[Variable]
        :param verbose: whether to print iterations, defaults to 0
        :type verbose: int, optional
        :param constraints: a list of natural language constraints, defaults to []
        :type constraints: List[str], optional
        :param optimizer_system_prompt: system prompt to the optimizer, defaults to textgrad.prompts.OPTIMIZER_SYSTEM_PROMPT. Needs to accept new_variable_start_tag and new_variable_end_tag
        :type optimizer_system_prompt: str, optional
        :param in_context_examples: a list of in-context examples, defaults to []
        :type in_context_examples: List[str], optional
        :param gradient_memory: the number of past gradients to store, defaults to 0
        :type gradient_memory: int, optional
        """
        super().__init__(parameters, engine=engine, verbose=verbose, constraints=constraints, new_variable_tags=new_variable_tags, optimizer_system_prompt=optimizer_system_prompt, in_context_examples=in_context_examples, gradient_memory=gradient_memory)
        assert isinstance(self.engine, GuidanceEngine), "GuidedTextualGradientDescent optimizer requires a GuidanceEngine engine. Got: {}".format(self.engine)

    def step(self):
        """
        Perform a single optimization step.
        This method updates the parameters of the optimizer by generating new text using the engine and updating the parameter values accordingly.
        It also logs the optimizer response and the updated text.
        Returns:
            None
        """
        for parameter in self.parameters:
            prompt_update_parameter = self._update_prompt(parameter)
            # Change the below with the guidance function
            @guidance
            def structured_tgd_response(lm, 
                                        tgd_prompt: str=prompt_update_parameter, 
                                        system_prompt: str=self.optimizer_system_prompt,
                                        new_variable_tags: List[str]=self.new_variable_tags, 
                                        max_reasoning_tokens: int=1024, 
                                        max_variable_tokens: int=4096):
                with guidance.system():
                    lm += system_prompt
                with guidance.user():
                    lm += tgd_prompt
                with guidance.assistant():
                    lm += "Reasoning: " + gen(name="reasoning", stop="\n", max_tokens=max_reasoning_tokens) + "\n"
                    lm += new_variable_tags[0] + gen(name="improved_variable", stop=new_variable_tags[1], max_tokens=max_variable_tokens) + new_variable_tags[1]
                return lm
            structured_response = self.engine.generate_structured(structured_tgd_response, tgd_prompt=prompt_update_parameter)
            new_value = structured_response["improved_variable"]
            logger.info(f"GuidedTextualGradientDescent output variables", extra={"optimizer.response": structured_response})
            logger.info(f"GuidedTextualGradientDescent optimizer response", extra={"optimizer.response": new_value})
            parameter.set_value(new_value)
            logger.info(f"GuidedTextualGradientDescent updated text", extra={"parameter.value": parameter.value})
            if self.verbose:
                print("-----------------------GuidedTextualGradientDescent------------------------")
                print(parameter.value)
            
            if self.do_gradient_memory:
                self.update_gradient_memory(parameter)
