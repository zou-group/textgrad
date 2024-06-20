import logging
from typing import List, Union

from textgrad.engine import EngineLM, get_engine
from textgrad.variable import Variable
from textgrad.autograd import LLMCall, FormattedLLMCall, Module
from .config import SingletonBackwardEngine

logging.disable(logging.CRITICAL)

DEFAULT_TEST_TIME = (
    "You are an intelligent assistant used as an evaluator, and part of an optimization system. "
    "You will analyze a solution to a multi-choice problem. Investigate the reasoning and answer. "
    "Do not try to solve the problem, only raise the potential issues and mistakes in the answer. "
    "Be creative, think about different perspectives, and be very critical."
)


class TextLoss(Module):
    """
    A vanilla loss function to evaluate a response using an LLM.
    """

    def __init__(self, eval_system_prompt: Union[Variable, str], engine: Union[EngineLM, str] = None):
        super().__init__()

        if isinstance(eval_system_prompt, str):
            eval_system_prompt = Variable(
                eval_system_prompt,
                requires_grad=False,
                role_description="system prompt for the evaluation"
            )

        self.eval_system_prompt = eval_system_prompt

        if engine is None:
            engine = SingletonBackwardEngine().get_engine()
            if engine is None:
                raise Exception(
                    "No engine provided. Either provide an engine as the argument to this call, or use "
                    "`textgrad.set_backward_engine(engine)` to set the backward engine."
                )

        if isinstance(engine, str):
            engine = get_engine(engine)

        self.engine = engine
        self.llm_call = LLMCall(self.engine, self.eval_system_prompt)

    def forward(self, instance: Variable) -> Variable:
        """
        Calls the LLM for response evaluation.

        :param instance: The instance variable to be evaluated.
        :return: The result of the evaluation.
        """
        return self.llm_call(instance)


class MultiFieldEvaluation(Module):
    """
    A module to compare two variables using a language model.
    """

    def __init__(
        self,
        evaluation_instruction: Variable,
        role_descriptions: List[str],
        engine: Union[EngineLM, str] = None,
        system_prompt: Variable = None
    ):
        super().__init__()

        self.evaluation_instruction = evaluation_instruction

        if engine is None:
            engine = SingletonBackwardEngine().get_engine()
            if engine is None:
                raise Exception(
                    "No engine provided. Either provide an engine as the argument to this call, or use "
                    "`textgrad.set_backward_engine(engine)` to set the backward engine."
                )

        if isinstance(engine, str):
            engine = get_engine(engine)

        self.engine = engine
        self.role_descriptions = role_descriptions

        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = Variable(
                "You are an evaluation system that compares two variables.",
                requires_grad=False,
                role_description="system prompt for the evaluation"
            )

        format_string_items = ["{instruction}"]
        format_string_items += [f"**{role}**: {{{role}}}" for role in role_descriptions]
        self.format_string = "\n".join(format_string_items).format(instruction=self.evaluation_instruction)

        self.fields = {"instruction": self.evaluation_instruction}
        self.fields.update({role: None for role in role_descriptions})

        self.formatted_llm_call = FormattedLLMCall(
            engine=self.engine,
            format_string=self.format_string,
            fields=self.fields,
            system_prompt=self.system_prompt
        )

    def forward(self, inputs: List[Variable]) -> Variable:
        """
        Performs the evaluation of multiple fields.

        :param inputs: List of variables to be evaluated.
        :return: The result of the evaluation.
        """
        for role, var in zip(self.role_descriptions, inputs):
            var.set_role_description(role)

        inputs_call = {"instruction": self.evaluation_instruction}
        inputs_call.update({role: var for role, var in zip(self.role_descriptions, inputs)})

        return self.formatted_llm_call(inputs=inputs_call, response_role_description="evaluation of the prediction")


class MultiFieldTokenParsedEvaluation(MultiFieldEvaluation):
    """
    A subclass of MultiFieldEvaluation that parses the output response.
    """

    def __init__(
        self,
        evaluation_instruction: Variable,
        role_descriptions: List[str],
        engine: Union[EngineLM, str] = None,
        system_prompt: Variable = None,
        parse_tags: List[str] = None
    ):
        super().__init__(evaluation_instruction, role_descriptions, engine, system_prompt)
        self.parse_tags = parse_tags

    def parse_output(self, response: Variable) -> str:
        """
        Parses the output response and returns the parsed response.

        :param response: The response to be parsed.
        :return: The parsed response.
        """
        response_text = response.value
        parsed_response = response_text.split(self.parse_tags[0])[1].split(self.parse_tags[1])[0].strip()
        return parsed_response


class MultiChoiceTestTime(Module):
    """
    The test-time loss to use when working on a response to a multiple-choice question.
    """

    def __init__(self, evaluation_instruction: str, engine: Union[EngineLM, str] = None, system_prompt: Variable = None):
        super().__init__()

        if system_prompt:
            self.tt_system_prompt = system_prompt
        else:
            self.tt_system_prompt = Variable(
                DEFAULT_TEST_TIME,
                requires_grad=False,
                role_description="system prompt for the test-time evaluation"
            )

        if engine is None:
            engine = SingletonBackwardEngine().get_engine()
            if engine is None:
                raise Exception(
                    "No engine provided. Either provide an engine as the argument to this call, or use "
                    "`textgrad.set_backward_engine(engine)` to set the backward engine."
                )

        if isinstance(engine, str):
            engine = get_engine(engine)

        self.engine = engine
        self.format_string = "{instruction}\nQuestion: {{question}}\nAnswer by the language model: {{prediction}}"
        self.format_string = self.format_string.format(instruction=evaluation_instruction)
        self.fields = {"prediction": None, "question": None}
        self.formatted_llm_call = FormattedLLMCall(
            engine=self.engine,
            format_string=self.format_string,
            fields=self.fields,
            system_prompt=self.tt_system_prompt
        )

    def forward(self, question: str, prediction: Variable) -> Variable:
        """
        Performs the evaluation of a multiple-choice question prediction.

        :param question: The question to be evaluated.
        :param prediction: The predicted answer variable.
        :return: The result of the evaluation.
        """
        question_variable = Variable(question, requires_grad=False, role_description="the multiple choice question")
        inputs = {"prediction": prediction, "question": question_variable}
        return self.formatted_llm_call(inputs=inputs, response_role_description=f"evaluation of the {prediction.get_role_description()}")
