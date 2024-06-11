from typing import List

from textgrad import Variable
from textgrad.engine import EngineLM
from textgrad.autograd.algebra import Sum, Aggregate
from textgrad.autograd.llm_ops import LLMCall, FormattedLLMCall

def sum(variables: List[Variable]) -> Variable:
    """
    Represents a sum operation on a list of variables. 
    In TextGrad, sum is simply concatenation of the values of the variables.

    :param variables: The list of variables to be summed (concatenated).
    :type variables: List[Variable]
    :return: A new variable representing the sum of the input variables.
    :rtype: Variable
    """
    return Sum()(variables)


def aggregate(variables: List[Variable]) -> Variable:
    """
    WIP - Aggregates a list of variables.
    In TextGrad, forward pass of aggregation is simply concatenation of the values of the variables.
    The backward pass performs a reduction operation on the gradients of the variables.
    This reduction is currently an LLM call to summarize the gradients.

    :param variables: The list of variables to be aggregated.
    :type variables: List[Variable]
    :return: The aggregated variable.
    :rtype: Variable
    """
    return Aggregate()(variables)


def llm_call(input_variable: Variable, engine: EngineLM, 
             response_role_description: str = None, system_prompt: Variable = None):
    """A functional version of the LLMCall.
    The simple LLM call function. This function will call the LLM with the input and return the response, also register the grad_fn for backpropagation.
    
    :param input_variable: The input variable (aka prompt) to use for the LLM call.
    :type input_variable: Variable
    :param response_role_description: Role description for the LLM response, defaults to VARIABLE_OUTPUT_DEFAULT_ROLE
    :type response_role_description: str, optional
    :return: response sampled from the LLM
    :rtype: Variable
    :param engine: engine to use for the LLM call
    :type engine: EngineLM
    :param input_role_description: role description for the input variable, defaults to VARIABLE_INPUT_DEFAULT_ROLE
    :type input_role_description: str, optional
    :param system_prompt: system prompt to use for the LLM call, default depends on the engine.
    :type system_prompt: Variable, optional
    
    >>> from textgrad import Variable, get_engine
    >>> from textgrad.autograd.functional import llm_call
    >>> engine = get_engine("gpt-3.5-turbo")
    >>> prompt = Variable("What is the capital of France?", role_description="prompt to the LM")
    >>> response = llm_call(prompt, engine=engine) 
    # This returns something like Variable(data=The capital of France is Paris., grads=)
    
    """
    return LLMCall(engine=engine, system_prompt=system_prompt)(input_variable, response_role_description)


def formatted_llm_call(inputs: List[Variable], response_role_description: str, 
                       engine: EngineLM, format_string: str, 
                       fields: dict[str, str], system_prompt: Variable = None):
    """A functional version of the LLM call with formatted strings. 
    Just a wrapper around the FormattedLLMCall class.
    
    This function will call the LLM with the input and return the response, also register the grad_fn for backpropagation.

    :param inputs: Variables to use for the input. This should be a mapping of the fields to the variables.
    :type inputs: dict[str, Variable]
    :param response_role_description: Role description for the response variable, defaults to VARIABLE_OUTPUT_DEFAULT_ROLE
    :type response_role_description: str, optional
    :param engine: The engine to use for the LLM call.
    :type engine: EngineLM
    :param format_string: The format string to use for the input. For instance, "The capital of {country} is {capital}". For a format string like this, we'll expect to have the fields dictionary to have the keys "country" and "capital". Similarly, in the forward pass, we'll expect the input variables to have the keys "country" and "capital".
    :type format_string: str
    :param fields: The fields to use for the format string. For the above example, this would be {"country": {}, "capital": {}}. This is currently a dictionary in case we'd want to inject more information later on.
    :type fields: dict[str, str]
    :param system_prompt: The system prompt to use for the LLM call. Default value depends on the engine.
    :type system_prompt: Variable, optional
    :return: Sampled response from the LLM
    :rtype: Variable
    """
    call_object = FormattedLLMCall(engine=engine, format_string=format_string, fields=fields, system_prompt=system_prompt)
    return call_object(inputs, response_role_description)