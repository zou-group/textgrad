from textgrad import logger
from textgrad.defaults import (SYSTEM_PROMPT_DEFAULT_ROLE, 
                               VARIABLE_OUTPUT_DEFAULT_ROLE)
from textgrad.variable import Variable
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default
from typing import List
from .llm_backward_prompts import (
    EVALUATE_VARIABLE_INSTRUCTION,
    CONVERSATION_START_INSTRUCTION_BASE,
    CONVERSATION_START_INSTRUCTION_CHAIN,
    CONVERSATION_TEMPLATE,
    OBJECTIVE_INSTRUCTION_CHAIN,
    OBJECTIVE_INSTRUCTION_BASE,
    BACKWARD_SYSTEM_PROMPT,
    IN_CONTEXT_EXAMPLE_PROMPT_ADDITION,
)
from .function import Function, BackwardContext


class LLMCall(Function):
    def __init__(self, engine: EngineLM, system_prompt: Variable = None):
        """The simple LLM call function. This function will call the LLM with the input and return the response, also register the grad_fn for backpropagation.

        :param engine: engine to use for the LLM call
        :type engine: EngineLM
        :param system_prompt: system prompt to use for the LLM call, default depends on the engine.
        :type system_prompt: Variable, optional
        """
        super().__init__()
        self.engine = validate_engine_or_get_default(engine)
        self.system_prompt = system_prompt
        if self.system_prompt and self.system_prompt.get_role_description() is None:
            self.system_prompt.set_role_description(SYSTEM_PROMPT_DEFAULT_ROLE)
    
    def forward(self, input_variable: Variable, response_role_description: str = VARIABLE_OUTPUT_DEFAULT_ROLE) -> Variable:
        """
        The LLM call. This function will call the LLM with the input and return the response, also register the grad_fn for backpropagation.
        
        :param input_variable: The input variable (aka prompt) to use for the LLM call.
        :type input_variable: Variable
        :param response_role_description: Role description for the LLM response, defaults to VARIABLE_OUTPUT_DEFAULT_ROLE
        :type response_role_description: str, optional
        :return: response sampled from the LLM
        :rtype: Variable
        
        :example:
        >>> from textgrad import Variable, get_engine
        >>> from textgrad.autograd.llm_ops import LLMCall
        >>> engine = get_engine("gpt-3.5-turbo")
        >>> llm_call = LLMCall(engine)
        >>> prompt = Variable("What is the capital of France?", role_description="prompt to the LM")
        >>> response = llm_call(prompt, engine=engine) 
        # This returns something like Variable(data=The capital of France is Paris., grads=)
        """
        # TODO: Should we allow default roles? It will make things less performant.
        system_prompt_value = self.system_prompt.value if self.system_prompt else None

        # Make the LLM Call
        response_text = self.engine(input_variable.value, system_prompt=system_prompt_value)

        # Create the response variable
        response = Variable(
            value=response_text,
            predecessors=[self.system_prompt, input_variable] if self.system_prompt else [input_variable],
            role_description=response_role_description
        )
        
        logger.info(f"LLMCall function forward", extra={"text": f"System:{system_prompt_value}\nQuery: {input_variable.value}\nResponse: {response_text}"})
        
        # Populate the gradient function, using a container to store the backward function and the context
        response.set_grad_fn(BackwardContext(backward_fn=self.backward, 
                                             response=response, 
                                             prompt=input_variable.value, 
                                             system_prompt=system_prompt_value))

        return response
    
    def backward(self, response: Variable, prompt: str, system_prompt: str, backward_engine: EngineLM):
        """
        Backward pass through the LLM call. This will register gradients in place.

        :param response: The response variable.
        :type response: Variable
        :param prompt: The prompt string that will be used as input to an LM.
        :type prompt: str
        :param system_prompt: The system prompt string.
        :type system_prompt: str
        :param backward_engine: The backward engine that will do the gradient computation.
        :type backward_engine: EngineLM

        :return: None
        """
        children_variables = response.predecessors
        if response.get_gradient_text() == "":
            self._backward_through_llm_base(children_variables, response, prompt, system_prompt, backward_engine)
        else:
            self._backward_through_llm_chain(children_variables, response, prompt, system_prompt, backward_engine)

    @staticmethod
    def _construct_llm_chain_backward_prompt(backward_info: dict[str, str]) -> str:
        conversation = CONVERSATION_TEMPLATE.format(**backward_info)
        backward_prompt = CONVERSATION_START_INSTRUCTION_CHAIN.format(conversation=conversation, **backward_info)
        backward_prompt += OBJECTIVE_INSTRUCTION_CHAIN.format(**backward_info)
        backward_prompt += EVALUATE_VARIABLE_INSTRUCTION.format(**backward_info)
        return backward_prompt

    @staticmethod
    def _backward_through_llm_chain(variables: List[Variable], 
                                    response: Variable, 
                                    prompt: str, 
                                    system_prompt: str,
                                    backward_engine: EngineLM):

        """
        Backward through the LLM to compute gradients for each variable, in the case where the output has gradients on them.
        i.e. applying the chain rule.
        
        :param variables: The list of variables to compute gradients for.
        :type variables: List[Variable]
        :param response: The response variable.
        :type response: Variable
        :param prompt: The prompt string.
        :type prompt: str
        :param system_prompt: The system prompt string.
        :type system_prompt: str
        :param backward_engine: The backward engine to use for computing gradients.
        :type backward_engine: EngineLM

        :return: None
        """
        for variable in variables:
            if not variable.requires_grad:
                continue

            backward_info = {
                "response_desc": response.get_role_description(),
                "response_value": response.get_value(),
                "response_gradient": response.get_gradient_text(),
                "prompt": prompt,
                "system_prompt": system_prompt,
                "variable_desc": variable.get_role_description(),
                "variable_short": variable.get_short_value()
            }
            
            backward_prompt = LLMCall._construct_llm_chain_backward_prompt(backward_info)

            logger.info(f"_backward_through_llm prompt", extra={"_backward_through_llm": backward_prompt})
            gradient_value = backward_engine(backward_prompt, system_prompt=BACKWARD_SYSTEM_PROMPT)
            logger.info(f"_backward_through_llm gradient", extra={"_backward_through_llm": gradient_value})
            
            var_gradients = Variable(value=gradient_value, role_description=f"feedback to {variable.get_role_description()}")
            variable.gradients.add(var_gradients)
            conversation = CONVERSATION_TEMPLATE.format(**backward_info)
            variable.gradients_context[var_gradients] = {
                "context": conversation, 
                "response_desc": response.get_role_description(),
                "variable_desc": variable.get_role_description()
            }
            
            if response._reduce_meta:
                var_gradients._reduce_meta.extend(response._reduce_meta)
                variable._reduce_meta.extend(response._reduce_meta)

    @staticmethod
    def _construct_llm_base_backward_prompt(backward_info: dict[str, str]) -> str:
        conversation = CONVERSATION_TEMPLATE.format(**backward_info)
        backward_prompt = CONVERSATION_START_INSTRUCTION_BASE.format(conversation=conversation, **backward_info)
        backward_prompt += OBJECTIVE_INSTRUCTION_BASE.format(**backward_info)
        backward_prompt += EVALUATE_VARIABLE_INSTRUCTION.format(**backward_info)
        return backward_prompt

    @staticmethod
    def _backward_through_llm_base(variables: List[Variable], 
                                   response: Variable,
                                   prompt: str,
                                   system_prompt: str,
                                   backward_engine: EngineLM):
        """
        Backward pass through the LLM base. 
        In this case we do not have gradients on the output variable.

        :param variables: A list of variables to compute gradients for.
        :type variables: List[Variable]
        :param response: The response variable.
        :type response: Variable
        :param prompt: The prompt string.
        :type prompt: str
        :param system_prompt: The system prompt string.
        :type system_prompt: str
        :param backward_engine: The backward engine to use for computing gradients.
        :type backward_engine: EngineLM
        """
        for variable in variables:
            if not variable.requires_grad:
                continue

            backward_info = {
                "response_desc": response.get_role_description(),
                "response_value": response.get_value(),
                "prompt": prompt,
                "system_prompt": system_prompt,
                "variable_desc": variable.get_role_description(),
                "variable_short": variable.get_short_value()
            }
            
            backward_prompt = LLMCall._construct_llm_base_backward_prompt(backward_info)
            
            logger.info(f"_backward_through_llm prompt", extra={"_backward_through_llm": backward_prompt})
            gradient_value = backward_engine(backward_prompt, system_prompt=BACKWARD_SYSTEM_PROMPT)
            logger.info(f"_backward_through_llm gradient", extra={"_backward_through_llm": gradient_value})

            conversation = CONVERSATION_TEMPLATE.format(**backward_info)
            var_gradients = Variable(value=gradient_value, role_description=f"feedback to {variable.get_role_description()}")
            variable.gradients.add(var_gradients)
            variable.gradients_context[var_gradients] = {
                "context": conversation, 
                "response_desc": response.get_role_description(),
                "variable_desc": variable.get_role_description()
            }

            if response._reduce_meta:
                var_gradients._reduce_meta.extend(response._reduce_meta)
                variable._reduce_meta.extend(response._reduce_meta)



class FormattedLLMCall(LLMCall):
    def __init__(self, 
                 engine: EngineLM, 
                 format_string: str,
                 fields: dict[str, str],
                 system_prompt: Variable = None):
        """This class is responsible for handling the formatting of the input before calling the LLM.
        It inherits from the LLMCall class and reuses its backward function.

        :param engine: The engine to use for the LLM call.
        :type engine: EngineLM
        :param format_string: The format string to use for the input. For instance, "The capital of {country} is {capital}". For a format string like this, we'll expect to have the fields dictionary to have the keys "country" and "capital". Similarly, in the forward pass, we'll expect the input variables to have the keys "country" and "capital".
        :type format_string: str
        :param fields: The fields to use for the format string. For the above example, this would be {"country": {}, "capital": {}}. This is currently a dictionary in case we'd want to inject more information later on.
        :type fields: dict[str, str]
        :param system_prompt: The system prompt to use for the LLM call. Default value depends on the engine.
        :type system_prompt: Variable, optional
        """
        super().__init__(engine, system_prompt)
        self.format_string = format_string
        self.fields = fields
    
    
    def forward(self, 
                inputs: dict[str, Variable], 
                response_role_description: str = VARIABLE_OUTPUT_DEFAULT_ROLE) -> Variable:
        """The LLM call with formatted strings. 
        This function will call the LLM with the input and return the response, also register the grad_fn for backpropagation.

        :param inputs: Variables to use for the input. This should be a mapping of the fields to the variables.
        :type inputs: dict[str, Variable]
        :param response_role_description: Role description for the response variable, defaults to VARIABLE_OUTPUT_DEFAULT_ROLE
        :type response_role_description: str, optional
        :return: Sampled response from the LLM
        :rtype: Variable
        """
        # First ensure that all keys are present in the fields
        assert set(inputs.keys()) == set(self.fields.keys()), f"Expected fields {self.fields.keys()} but got {inputs.keys()}"
        
        input_variables = list(inputs.values())
        
        # Now format the string
        formatted_input_string = self.format_string.format(**{k: inputs[k].value for k in inputs.keys()})

        # TODO: Should we allow default roles? It will make things less performant.
        system_prompt_value = self.system_prompt.value if self.system_prompt else None

        # Make the LLM Call
        response_text = self.engine(formatted_input_string, system_prompt=system_prompt_value)

        # Create the response variable
        response = Variable(
            value=response_text,
            predecessors=[self.system_prompt, *input_variables] if self.system_prompt else [*input_variables],
            role_description=response_role_description
        )
        
        logger.info(f"LLMCall function forward", extra={"text": f"System:{system_prompt_value}\nQuery: {formatted_input_string}\nResponse: {response_text}"})
        
        # Populate the gradient function, using a container to store the backward function and the context
        response.set_grad_fn(BackwardContext(backward_fn=self.backward, 
                                             response=response, 
                                             prompt=formatted_input_string, 
                                             system_prompt=system_prompt_value))
        
        return response


class LLMCall_with_in_context_examples(LLMCall):
    
    def forward(self, input_variable: Variable, response_role_description: str = VARIABLE_OUTPUT_DEFAULT_ROLE, in_context_examples: List[str]=None) -> Variable:
        """
        The LLM call. This function will call the LLM with the input and return the response, also register the grad_fn for backpropagation.
        
        :param input_variable: The input variable (aka prompt) to use for the LLM call.
        :type input_variable: Variable
        :param response_role_description: Role description for the LLM response, defaults to VARIABLE_OUTPUT_DEFAULT_ROLE
        :type response_role_description: str, optional
        :return: response sampled from the LLM
        :rtype: Variable
        
        :example:
        >>> from textgrad import Variable, get_engine
        >>> from textgrad.autograd.llm_ops import LLMCall
        >>> engine = get_engine("gpt-3.5-turbo")
        >>> llm_call = LLMCall(engine)
        >>> prompt = Variable("What is the capital of France?", role_description="prompt to the LM")
        >>> response = llm_call(prompt, engine=engine) 
        # This returns something like Variable(data=The capital of France is Paris., grads=)
        """
        # TODO: Should we allow default roles? It will make things less performant.
        system_prompt_value = self.system_prompt.value if self.system_prompt else None

        # Make the LLM Call
        response_text = self.engine(input_variable.value, system_prompt=system_prompt_value)


        # Create the response variable
        response = Variable(
            value=response_text.split('<FINAL>')[1].split('</FINAL>')[0].strip(),
            predecessors=[self.system_prompt, input_variable] if self.system_prompt else [input_variable],
            role_description=response_role_description
        )


        
        logger.info(f"LLMCall function forward", extra={"text": f"System:{system_prompt_value}\nQuery: {input_variable.value}\nResponse: {response_text}"})
        
        # Populate the gradient function, using a container to store the backward function and the context
        response.set_grad_fn(BackwardContext(backward_fn=self.backward, 
                                             response=response, 
                                             prompt=input_variable.value, 
                                             system_prompt=system_prompt_value,
                                             in_context_examples=in_context_examples))
        
        # Stopping creteria Could be improved, now is very ugly. 
        if "The plan doesn't need to be improved" in response_text.split('<FINAL>')[1].split('</FINAL>')[0].strip():
            response = None

        return response

    def backward(self, response: Variable, prompt: str, system_prompt: str, in_context_examples: List[str], backward_engine: EngineLM):
        """
        Backward pass through the LLM call. This will register gradients in place.

        :param response: The response variable.
        :type response: Variable
        :param prompt: The prompt string that will be used as input to an LM.
        :type prompt: str
        :param system_prompt: The system prompt string.
        :type system_prompt: str
        :param backward_engine: The backward engine that will do the gradient computation.
        :type backward_engine: EngineLM

        :return: None
        """
        children_variables = response.predecessors

        if response.get_gradient_text() == "":
            self._backward_through_llm_base(children_variables, response, prompt, system_prompt, backward_engine, in_context_examples)
        else:
            self._backward_through_llm_chain(children_variables, response, prompt, system_prompt, backward_engine, in_context_examples)

    @staticmethod
    def _construct_llm_chain_backward_prompt(backward_info: dict[str, str]) -> str:
        conversation = CONVERSATION_TEMPLATE.format(**backward_info)
        backward_prompt = CONVERSATION_START_INSTRUCTION_CHAIN.format(conversation=conversation, **backward_info)
        backward_prompt += OBJECTIVE_INSTRUCTION_CHAIN.format(**backward_info)
        if len(backward_info['in_context_examples']) > 0:
            backward_prompt += IN_CONTEXT_EXAMPLE_PROMPT_ADDITION.format(**backward_info)
        backward_prompt += EVALUATE_VARIABLE_INSTRUCTION.format(**backward_info)
       
        return backward_prompt
    @staticmethod
    def _backward_through_llm_chain(variables: List[Variable], 
                                    response: Variable, 
                                    prompt: str, 
                                    system_prompt: str,
                                    backward_engine: EngineLM,
                                    in_context_examples: List[str]=None):
        """
        Backward through the LLM to compute gradients for each variable, in the case where the output has gradients on them.
        i.e. applying the chain rule.
        
        :param variables: The list of variables to compute gradients for.
        :type variables: List[Variable]
        :param response: The response variable.
        :type response: Variable
        :param prompt: The prompt string.
        :type prompt: str
        :param system_prompt: The system prompt string.
        :type system_prompt: str
        :param backward_engine: The backward engine to use for computing gradients.
        :type backward_engine: EngineLM

        :return: None
        """
        for variable in variables:
            if not variable.requires_grad:
                continue

            backward_info = {
                "response_desc": response.get_role_description(),
                "response_value": response.get_value(),
                "response_gradient": response.get_gradient_text(),
                "prompt": prompt,
                "system_prompt": system_prompt,
                "variable_desc": variable.get_role_description(),
                "variable_short": variable.get_short_value(),
                "in_context_examples": "\n".join(in_context_examples) if in_context_examples is not None else [],
            }
            
            backward_prompt = LLMCall_with_in_context_examples._construct_llm_chain_backward_prompt(backward_info)

            logger.info(f"_backward_through_llm prompt", extra={"_backward_through_llm": backward_prompt})
            gradient_value = backward_engine(backward_prompt, system_prompt=BACKWARD_SYSTEM_PROMPT)
            logger.info(f"_backward_through_llm gradient", extra={"_backward_through_llm": gradient_value})
            
            var_gradients = Variable(value=gradient_value, role_description=f"feedback to {variable.get_role_description()}")
            variable.gradients.add(var_gradients)
            conversation = CONVERSATION_TEMPLATE.format(**backward_info)
            variable.gradients_context[var_gradients] = {
                "context": conversation, 
                "response_desc": response.get_role_description(),
                "variable_desc": variable.get_role_description()
            }
            
            if response._reduce_meta:
                var_gradients._reduce_meta.extend(response._reduce_meta)
                variable._reduce_meta.extend(response._reduce_meta)

    @staticmethod
    def _construct_llm_base_backward_prompt(backward_info: dict[str, str]) -> str:
        conversation = CONVERSATION_TEMPLATE.format(**backward_info)
        backward_prompt = CONVERSATION_START_INSTRUCTION_BASE.format(conversation=conversation, **backward_info)
        backward_prompt += OBJECTIVE_INSTRUCTION_BASE.format(**backward_info)
        
        if len(backward_info['in_context_examples']) > 0:
            backward_prompt += IN_CONTEXT_EXAMPLE_PROMPT_ADDITION.format(**backward_info)
        backward_prompt += EVALUATE_VARIABLE_INSTRUCTION.format(**backward_info)

        
        return backward_prompt

    @staticmethod
    def _backward_through_llm_base(variables: List[Variable], 
                                   response: Variable,
                                   prompt: str,
                                   system_prompt: str,
                                   backward_engine: EngineLM,
                                   in_context_examples: List[str]=None):
        """
        Backward pass through the LLM base. 
        In this case we do not have gradients on the output variable.

        :param variables: A list of variables to compute gradients for.
        :type variables: List[Variable]
        :param response: The response variable.
        :type response: Variable
        :param prompt: The prompt string.
        :type prompt: str
        :param system_prompt: The system prompt string.
        :type system_prompt: str
        :param backward_engine: The backward engine to use for computing gradients.
        :type backward_engine: EngineLM
        """

        for variable in variables:
            if not variable.requires_grad:
                continue

            backward_info = {
                "response_desc": response.get_role_description(),
                "response_value": response.get_value(),
                "prompt": prompt,
                "system_prompt": system_prompt,
                "variable_desc": variable.get_role_description(),
                "variable_short": variable.get_short_value(),
                "in_context_examples": "\n".join(in_context_examples) if in_context_examples is not None else [],

            }
            
            backward_prompt = LLMCall_with_in_context_examples._construct_llm_base_backward_prompt(backward_info)
            
            logger.info(f"_backward_through_llm prompt", extra={"_backward_through_llm": backward_prompt})
            gradient_value = backward_engine(backward_prompt, system_prompt=BACKWARD_SYSTEM_PROMPT)
            logger.info(f"_backward_through_llm gradient", extra={"_backward_through_llm": gradient_value})

            conversation = CONVERSATION_TEMPLATE.format(**backward_info)
            var_gradients = Variable(value=gradient_value, role_description=f"feedback to {variable.get_role_description()}")
            variable.gradients.add(var_gradients)
            variable.gradients_context[var_gradients] = {
                "context": conversation, 
                "response_desc": response.get_role_description(),
                "variable_desc": variable.get_role_description()
            }

            if response._reduce_meta:
                var_gradients._reduce_meta.extend(response._reduce_meta)
                variable._reduce_meta.extend(response._reduce_meta)