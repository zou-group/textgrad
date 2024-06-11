from abc import ABC, abstractmethod
from typing import List, Union
from collections import defaultdict
from textgrad.variable import Variable
from textgrad import logger
from textgrad.engine import EngineLM, get_engine
from .optimizer_prompts import construct_tgd_prompt, OPTIMIZER_SYSTEM_PROMPT
from textgrad.config import SingletonBackwardEngine

class Optimizer(ABC):
    """
    Base class for all optimizers.

    :param parameters: The list of parameters to optimize.
    :type parameters: List[Variable]

    :Methods:
        - zero_grad(): Clears the gradients of all parameters.
        - step(): Performs a single optimization step.
    """

    def __init__(self, parameters: List[Variable]):
        self.parameters = parameters

    def zero_grad(self):
        """
        Clears the gradients of all parameters.
        """
        for p in self.parameters:
            p.gradients = set()

    @abstractmethod
    def step(self):
        """
        Performs a single optimization step.
        """
        pass


class TextualGradientDescent(Optimizer):
    def __init__(self, 
                 parameters: List[Variable], 
                 verbose: int=0, 
                 engine: Union[EngineLM, str]=None, 
                 constraints: List[str]=None,
                 new_variable_tags: List[str]=["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"],
                 optimizer_system_prompt: str=OPTIMIZER_SYSTEM_PROMPT,
                 in_context_examples: List[str]=None,
                 gradient_memory: int=0):
        """TextualGradientDescent optimizer

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
        super().__init__(parameters)
        if ((engine is None) and (SingletonBackwardEngine().get_engine() is None)):
            raise Exception("No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
        elif engine is None:
            engine = SingletonBackwardEngine().get_engine()
        if isinstance(engine, str):
            engine = get_engine(engine)
        self.engine = engine
        self.verbose = verbose
        self.constraints = constraints if constraints is not None else []
        self.optimizer_system_prompt = optimizer_system_prompt.format(new_variable_start_tag=new_variable_tags[0], new_variable_end_tag=new_variable_tags[1])
        self.do_constrained = (len(self.constraints) > 0)
        self.new_variable_tags = new_variable_tags
        self.in_context_examples = in_context_examples if in_context_examples is not None else []
        self.do_in_context_examples = (len(self.in_context_examples) > 0)
        self.gradient_memory = gradient_memory
        self.gradient_memory_dict = defaultdict(list)
        self.do_gradient_memory = (gradient_memory > 0)

    @property
    def constraint_text(self):
        """
        Returns a formatted string representation of the constraints.

        :return: A string containing the constraints in the format "Constraint {index}: {constraint}".
        :rtype: str
        """
        constraints_ordered = [f"Constraint {i+1}: {constraint}" for i, constraint in enumerate(self.constraints)]
        return "\n".join(constraints_ordered)
    
    def get_gradient_memory_text(self, variable: Variable):
        grad_memory = ""
        variable_grad_memory = self.gradient_memory_dict[variable][-self.gradient_memory:]
        for i, grad_info in enumerate(variable_grad_memory):
            grad_memory += f"\n<FEEDBACK-{i+1}> {grad_info['value']}</FEEDBACK-{i+1}>\n"
        return grad_memory
    
    def update_gradient_memory(self, variable: Variable):
        self.gradient_memory_dict[variable].append({"value": variable.get_gradient_text()})
    
    def _update_prompt(self, variable: Variable):
        grad_memory = self.get_gradient_memory_text(variable)
        optimizer_information = {
            "variable_desc": variable.get_role_description(),
            "variable_value": variable.value,
            "variable_grad": variable.get_gradient_and_context_text(),
            "variable_short": variable.get_short_value(),
            "constraint_text": self.constraint_text,
            "new_variable_start_tag": self.new_variable_tags[0],
            "new_variable_end_tag": self.new_variable_tags[1],
            "in_context_examples": "\n".join(self.in_context_examples),
            "gradient_memory": grad_memory
        }
        
        prompt = construct_tgd_prompt(do_constrained=self.do_constrained, 
                                      do_in_context_examples=(self.do_in_context_examples and (len(self.in_context_examples) > 0)),
                                      do_gradient_memory=(self.do_gradient_memory and (grad_memory != "")),
                                      **optimizer_information)
        
        logger.info(f"TextualGradientDescent prompt for update", extra={"prompt": prompt})
        return prompt

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
            new_text = self.engine(prompt_update_parameter, system_prompt=self.optimizer_system_prompt)
            logger.info(f"TextualGradientDescent optimizer response", extra={"optimizer.response": new_text})
            parameter.set_value(new_text.split(self.new_variable_tags[0])[1].split(self.new_variable_tags[1])[0].strip())
            logger.info(f"TextualGradientDescent updated text", extra={"parameter.value": parameter.value})
            if self.verbose:
                print("-----------------------TextualGradientDescent------------------------")
                print(parameter.value)
            
            if self.do_gradient_memory:
                self.update_gradient_memory(parameter)


class TextualGradientDescentwithMomentum(Optimizer):
    def __init__(self, 
                 engine: Union[str, EngineLM], 
                 parameters: List[Variable], 
                 momentum_window: int = 0, 
                 constraints: List[str]=None,
                 new_variable_tags: List[str]=["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"],
                 in_context_examples: List[str]=None,
                 optimizer_system_prompt: str=OPTIMIZER_SYSTEM_PROMPT):
        super().__init__(parameters)
        if ((engine is None) and (SingletonBackwardEngine().get_engine() is None)):
            raise Exception("No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
        elif engine is None:
            engine = SingletonBackwardEngine().get_engine()
        if isinstance(engine, str):
            engine = get_engine(engine)
        self.engine = engine
        
        if momentum_window == 0:
            return TextualGradientDescent(engine=engine, parameters=parameters, constraints=constraints)
        # Each item in the momentum storage will include past value and the criticsm
        self.momentum_storage = [[] for _ in range(len(parameters))]
        self.momentum_window = momentum_window
        self.do_momentum = True
        self.constraints = constraints if constraints is not None else []
        self.do_constrained = (len(self.constraints) > 0)
        self.optimizer_system_prompt = optimizer_system_prompt.format(new_variable_start_tag=new_variable_tags[0], new_variable_end_tag=new_variable_tags[1])
        self.new_variable_tags = new_variable_tags
        self.in_context_examples = in_context_examples if in_context_examples is not None else []
        self.do_in_context_examples = (len(self.in_context_examples) > 0)

        logger.info(f"TextualGradientDescent initialized with momentum window: {momentum_window}")

    @property
    def constraint_text(self):
        constraints_ordered = [f"Constraint {i+1}: {constraint}" for i, constraint in enumerate(self.constraints)]
        return "\n".join(constraints_ordered)
    
    def _update_prompt(self, variable: Variable, momentum_storage_idx: int):
        past_values = ""
        
        past_n_steps = self.momentum_storage[momentum_storage_idx]
        for i, step_info in enumerate(past_n_steps):
            past_values += f"\n{variable.get_role_description()} at Step {i + 1}: {step_info['value']}.\n"

        optimizer_information = {
            "variable_desc": variable.get_role_description(),
            "variable_value": variable.value,
            "variable_grad": variable.get_gradient_text(),
            "variable_short": variable.get_short_value(),
            "constraint_text": self.constraint_text,
            "past_values": past_values,
            "new_variable_start_tag": self.new_variable_tags[0],
            "new_variable_end_tag": self.new_variable_tags[1],
            "in_context_examples": "\n".join(self.in_context_examples)
        }
        
        prompt = construct_tgd_prompt(do_momentum=(self.do_momentum and (past_values != "")), 
                                      do_constrained=self.do_constrained, 
                                      do_in_context_examples=(self.do_in_context_examples and (len(self.in_context_examples) > 0)),
                                      **optimizer_information)
        
        logger.info(f"TextualGradientwithMomentum prompt for update", extra={"prompt": prompt})


    def _update_momentum_storage(self, variable: Variable, momentum_storage_idx: int):
        if len(self.momentum_storage[momentum_storage_idx]) >= self.momentum_window:
            self.momentum_storage[momentum_storage_idx].pop(0)
        
        self.momentum_storage[momentum_storage_idx].append({"value": variable.value, "gradients": variable.get_gradient_and_context_text()})
        
    def step(self):
        for idx, parameter in enumerate(self.parameters):
            self._update_momentum_storage(parameter, momentum_storage_idx=idx)
            prompt_update_parameter = self._update_prompt(parameter, momentum_storage_idx=idx)
            new_text = self.engine(prompt_update_parameter, system_prompt=self.optimizer_system_prompt)
            logger.info(f"TextualGradientDescentwithMomentum optimizer response", extra={"optimizer.response": new_text})
            parameter.set_value(new_text.split(self.new_variable_tags[0])[1].split(self.new_variable_tags[1])[0].strip())
            logger.info(f"TextualGradientDescentwithMomentum updated text", extra={"parameter.value": parameter.value})
