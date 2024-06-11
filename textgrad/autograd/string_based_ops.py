from textgrad import logger
from textgrad.variable import Variable
from textgrad.engine import EngineLM
from .function import Function, BackwardContext
from typing import Callable, Dict, List

CONVERSATION_TEMPLATE_STRING = (
    "Function purpose: {function_purpose}\n\n"
    "<INPUTS_TO_FUNCTION> {inputs_string} </INPUTS_TO_FUNCTION>\n\n"
    "<OUTPUT_OF_FUNCTION> {response_value} </OUTPUT_OF_FUNCTION>\n\n"
)

# Has the gradient on the output.
CONVERSATION_START_INSTRUCTION_STRING_FN_CHAIN = (
    "You will give feedback to a variable with the following role: <ROLE> {variable_desc} </ROLE>. "
    "Here is an evaluation of a string-based function with inputs and outputs :\n\n"
    "{conversation}"
)

# Does not have gradient on the output
CONVERSATION_START_INSTRUCTION_STRING_FN_BASE = (
    "You will give feedback to a variable with the following role: <ROLE> {variable_desc} </ROLE>. "
    "Here is an evaluation of the variable using a string-based function:\n\n"
    "{conversation}"
)

OBJECTIVE_INSTRUCTION_CHAIN = (
    "This conversation is part of a larger system. The <OUTPUT_OF_FUNCTION> was later used as {response_desc}.\n\n"
    "<OBJECTIVE_FUNCTION>Your goal is to give feedback to the variable to address the following feedback on the OUTPUT_OF_FUNCTION: {response_gradient} </OBJECTIVE_FUNCTION>\n\n"
)

OBJECTIVE_INSTRUCTION_BASE = (
    "<OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism to the variable given the above evaluation output. "
    "Our only goal is to improve the above metric, and nothing else. </OBJECTIVE_FUNCTION>\n\n"
)

# Some instructions for the backward pass are shared with LLMs
from .llm_backward_prompts import (
    EVALUATE_VARIABLE_INSTRUCTION,
    BACKWARD_SYSTEM_PROMPT
)

class StringBasedFunction(Function):
    def __init__(self, fn: Callable, function_purpose: str):
        """
        Autograd function for string-based functions.

        :param fn: The function to execute for the forward pass.
        :type fn: Callable
        :param function_purpose: The description of the purpose of the function. Analogous to role description for variables.
        :type function_purpose: str
        """
        super().__init__()
        self.fn = fn
        self.function_purpose = function_purpose

    def forward(self, 
                inputs: Dict[str, Variable], 
                response_role_description: str = None) -> Variable:
        """
        The forward mode for string-based functions

        :param inputs: The arguments that will be passed to the string based function. The keys are the names of the arguments.
        :type fn: Dict[str, Variable]
        :param response_role_description: The role description of the output variable. 
        :type response_role_description: str
        """        
        if response_role_description is None:
            response_role_description = f"Output of the string-based function with purpose: {self.function_purpose}"
        response_string = self.fn(**inputs)

        # Create the response variable
        response = Variable(
            value=response_string,
            predecessors=list(inputs.values()),
            role_description=response_role_description
        )
        
        logger.info(f"StringBasedFunction", extra={"text": f"In: {inputs}, Out: {response_string}"})
        
        # Populate the gradient function, using a container to store the backward function and the context
        response.set_grad_fn(BackwardContext(backward_fn=self.backward, 
                                             response=response, 
                                             function_purpose=self.function_purpose,
                                             inputs=inputs))

        return response
    
    def backward(self, response: Variable, 
                 function_purpose: str, 
                 inputs: Dict[str, Variable], 
                 backward_engine: EngineLM):
        children_variables = response.predecessors
        if response.get_gradient_text().strip() == "":
            self._backward_through_string_fn_base(children_variables, response, inputs, function_purpose, backward_engine)
        else:
            self._backward_through_string_fn_chain(children_variables, response, inputs, function_purpose, backward_engine)

    @staticmethod
    def _construct_string_fn_chain_backward_prompt(backward_info: dict[str, str]) -> str:
        conversation = CONVERSATION_TEMPLATE_STRING.format(**backward_info)
        backward_prompt = CONVERSATION_START_INSTRUCTION_STRING_FN_CHAIN.format(conversation=conversation, **backward_info)
        backward_prompt += OBJECTIVE_INSTRUCTION_CHAIN.format(**backward_info)
        backward_prompt += EVALUATE_VARIABLE_INSTRUCTION.format(**backward_info)
        return backward_prompt

    @staticmethod
    def _backward_through_string_fn_chain(variables: List[Variable],
                                          response: Variable,
                                          inputs: Dict[str, Variable],
                                          function_purpose: str, 
                                          backward_engine: EngineLM):
        inputs_string = "\n\n".join([f"**{k.replace('_', ' ').capitalize()}(role: {v.get_role_description()})**: {v.get_short_value()}" for k, v in inputs.items()])
        
        for variable in variables:
            if not variable.requires_grad:
                continue

            backward_info = {
                "response_desc": response.get_role_description(),
                "response_value": response.get_value(),
                "response_gradient": response.get_gradient_text(),
                "function_purpose": function_purpose,
                "inputs_string": inputs_string,
                "variable_desc": variable.get_role_description(),
                "variable_short": variable.get_short_value()
            }
            
            backward_prompt = StringBasedFunction._construct_string_fn_chain_backward_prompt(backward_info)

            logger.info(f"_backward_through_string_fn", extra={"_backward_through_string_fn": backward_prompt})
            gradient_value = backward_engine(backward_prompt, system_prompt=BACKWARD_SYSTEM_PROMPT)
            logger.info(f"_backward_through_string_fn gradient", extra={"_backward_through_string_fn": gradient_value})
            
            var_gradients = Variable(value=gradient_value, role_description=f"feedback to {variable.get_role_description()}")
            variable.gradients.add(var_gradients)
            conversation = CONVERSATION_TEMPLATE_STRING.format(**backward_info)
            variable.gradients_context[var_gradients] = {
                "context": conversation, 
                "response_desc": response.get_role_description(),
                "variable_desc": variable.get_role_description()
            }
            
            if response._reduce_meta:
                var_gradients._reduce_meta.extend(response._reduce_meta)
                variable._reduce_meta.extend(response._reduce_meta)

    @staticmethod
    def _construct_string_fn_base_backward_prompt(backward_info: dict[str, str]) -> str:
        conversation = CONVERSATION_TEMPLATE_STRING.format(**backward_info)
        backward_prompt = CONVERSATION_START_INSTRUCTION_STRING_FN_BASE.format(conversation=conversation, **backward_info)
        backward_prompt += OBJECTIVE_INSTRUCTION_BASE.format(**backward_info)
        backward_prompt += EVALUATE_VARIABLE_INSTRUCTION.format(**backward_info)
        return backward_prompt

    @staticmethod
    def _backward_through_string_fn_base(variables: List[Variable],
                                         response: Variable,
                                         inputs: Dict[str, Variable],
                                         function_purpose: str, 
                                         backward_engine: EngineLM):
        inputs_string = "\n\n".join([f"**{k.replace('_', ' ').capitalize()}(role: {v.get_role_description()})**: {v.get_short_value()}" for k, v in inputs.items()])
        
        for variable in variables:
            if not variable.requires_grad:
                continue

            backward_info = {
                "response_desc": response.get_role_description(),
                "response_value": response.get_value(),
                "response_gradient": response.get_gradient_text(),
                "function_purpose": function_purpose,
                "inputs_string": inputs_string,
                "variable_desc": variable.get_role_description(),
                "variable_short": variable.get_short_value()
            }
            backward_prompt = StringBasedFunction._construct_string_fn_base_backward_prompt(backward_info)
            
            logger.info(f"_backward_through_string_fn prompt", extra={"_backward_through_string_fn": backward_prompt})
            gradient_value = backward_engine(backward_prompt, system_prompt=BACKWARD_SYSTEM_PROMPT)
            logger.info(f"_backward_through_string_fn gradient", extra={"_backward_through_string_fn": gradient_value})

            conversation = CONVERSATION_TEMPLATE_STRING.format(**backward_info)
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
