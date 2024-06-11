## Operations over variables.
from typing import List, Set
from textgrad import logger
from textgrad.variable import Variable
from textgrad.engine import EngineLM
from .reduce_prompts import construct_reduce_prompt, REDUCE_MEAN_SYSTEM_PROMPT
from .function import Function, BackwardContext


def _reduce_gradients_mean(gradients: Set[Variable], backward_engine: EngineLM) -> Variable:
    """A function to reduce gradients by taking the "mean" of the gradients. 
    In this case, we use a summarization model to summarize the gradients.
    This can be helpful in batch training, where we want to average the gradients over a batch.

    :param gradients: Gradients to summarize together
    :type gradients: Set[Variable]
    :param backward_engine: The engine to summarize gradients. 
    :type backward_engine: EngineLM
    :return: The reduce(summarized) gradients
    :rtype: Variable
    """
    gradient_reduce_prompt = construct_reduce_prompt(gradients)
    reduced_gradients = backward_engine(gradient_reduce_prompt, system_prompt=REDUCE_MEAN_SYSTEM_PROMPT)
    gradient_descriptions = set([g.get_role_description() for g in gradients])
    gradient_descriptions = ", ".join(gradient_descriptions)
    reduced_gradients_variable = Variable(reduced_gradients, role_description=gradient_descriptions)
    logger.info(f"Reduced gradients", extra={"reduced_gradients": reduced_gradients_variable.value})
    # TODO: We need to add context for these gradients
    # Otherwise, .get_gradient_and_context_text() will return an empty string
    logger.info(f"Reduced gradients", extra={"reduced_gradients": reduced_gradients_variable.value})
    return reduced_gradients_variable


class Sum(Function):
    """
    Represents a sum operation on a list of variables. 
    In TextGrad, sum is simply concatenation of the values of the variables.

    :param variables: The list of variables to be summed (concatenated).
    :type variables: List[Variable]
    :return: A new variable representing the sum of the input variables.
    :rtype: Variable
    """

    def forward(self, variables: List[Variable]) -> Variable:
        """
        Performs the forward pass of the sum (concatenation) operation.

        :param variables: The list of variables to be summed.
        :type variables: List[Variable]
        :return: A new variable representing the sum of the input variables.
        :rtype: Variable
        """
        concat_values = "\n".join([v.get_value() for v in variables])
        role_descriptions = set([v.get_role_description() for v in variables])
        role_descriptions = ", ".join(role_descriptions)
        
        total = Variable(
            value=concat_values,
            predecessors=variables,
            role_description=f"a combination of the following: {role_descriptions}",
            requires_grad=any([v.requires_grad for v in variables]),
        )
        
        total.set_grad_fn(BackwardContext(backward_fn=self.backward,
                                          summation=total))
        
        return total

        
    def backward(self, summation: Variable, backward_engine: EngineLM):
        """
        Performs the backward pass of the sum operation.
        This is simply an idempotent operation, where we pass the feedback to the predecessors variables.

        :param summation: The variable representing the sum.
        :type summation: Variable
        :param backward_engine: The backward engine used for backpropagation.
        :type backward_engine: EngineLM
        """
        children_variables = summation.predecessors
        summation_gradients = summation.get_gradient_text()
        for variable in children_variables: 
            if summation_gradients == "":
                variable_gradient_value = ""
            else:
                variable_gradient_value = f"Here is the combined feedback we got for this specific {variable.get_role_description()} and other variables: {summation_gradients}."
                
            logger.info(f"Idempotent backward", extra={"v_gradient_value": variable_gradient_value, 
                                                    "summation_role": summation.get_role_description()})

            var_gradients = Variable(value=variable_gradient_value, 
                                    role_description=f"feedback to {variable.get_role_description()}")
            variable.gradients.add(var_gradients)
            
            if summation._reduce_meta != []:
                var_gradients._reduce_meta.extend(summation._reduce_meta)
                variable._reduce_meta.extend(summation._reduce_meta)

            variable.gradients.add(Variable(value=variable_gradient_value, 
                                            role_description=f"feedback to {variable.get_role_description()}"))


class Aggregate(Function):
    """This function is WIP
    """
    def forward(self, variables: List[Variable]) -> Variable:
        """
        Aggregates a list of variables.
        In TextGrad, forward pass of aggregation is simply concatenation of the values of the variables.
        The backward pass performs a reduction operation on the gradients of the variables.
        This reduction is currently an LLM call to summarize the gradients.

        :param variables: The list of variables to be aggregated.
        :type variables: List[Variable]
        :return: The aggregated variable.
        :rtype: Variable
        """
        concat_values = "\n".join([v.get_value() for v in variables])
        role_descriptions = set([v.get_role_description() for v in variables])
        role_descriptions = ", ".join(role_descriptions)
        
        # We create a unique meta tag that identifies which variables are aggregated together.
        # We also need to communicate to the variables that they are part of a mean operation.
        reduce_meta = {"op": _reduce_gradients_mean, "id": id(variables)}
        
        aggregated_variable = Variable(value=concat_values, 
                                    role_description=f"a combination of the following variables: {role_descriptions}.",
                                    predecessors=variables,
                                    requires_grad=any([v.requires_grad for v in variables]))
        
        aggregated_variable.set_grad_fn(BackwardContext(backward_fn=self.backward,
                                                        aggregated_variable=aggregated_variable))
        
        aggregated_variable._reduce_meta = [reduce_meta]
        return aggregated_variable
        
    def backward(self, aggregated_variable: Variable, backward_engine: EngineLM):
        children_variable = aggregated_variable.predecessors
        for variable in children_variable:
            aggregate_gradients = aggregated_variable.get_gradient_text()
        if aggregate_gradients == "":
            variable_gradient_value = ""
        else:
            variable_gradient_value = f"Here is the combined feedback we got for this specific {variable.get_role_description()} and other variables: {aggregate_gradients}."
            
        logger.info(f"aggregation backward", extra={"v_gradient_value": variable_gradient_value, 
                                                "aggregation_role": aggregated_variable.get_role_description()})

        var_gradients = Variable(value=variable_gradient_value, 
                                role_description=f"feedback to {variable.get_role_description()}")
        variable.gradients.add(var_gradients)
        
        if aggregated_variable._reduce_meta != []:
            var_gradients._reduce_meta.extend(aggregated_variable._reduce_meta)
            variable._reduce_meta.extend(aggregated_variable._reduce_meta)
