from textgrad.engine import EngineLM, get_engine
from textgrad.variable import Variable
from typing import List, Union
from textgrad.autograd import LLMCall, FormattedLLMCall, OrderedFieldsMultimodalLLMCall
from textgrad.autograd import Module
from .config import SingletonBackwardEngine


class TextLoss(Module):
    def __init__(self, 
                 eval_system_prompt: Union[Variable, str],
                 engine: Union[EngineLM, str] = None):
        """
        A vanilla loss function to evaluate a response.
        In particular, this module is used to evaluate any given text object.

        :param evaluation_instruction: The evaluation instruction variable.
        :type evaluation_instruction: Variable
        :param engine: The EngineLM object.
        :type engine: EngineLM
        
        :example:
        >>> from textgrad import get_engine, Variable
        >>> from textgrad.loss import TextLoss
        >>> engine = get_engine("gpt-4o")
        >>> evaluation_instruction = Variable("Is ths a good joke?", requires_grad=False)
        >>> response_evaluator = TextLoss(evaluation_instruction, engine)
        >>> response = Variable("What did the fish say when it hit the wall? Dam.", requires_grad=True)
        >>> response_evaluator(response)
        """
        super().__init__()
        if isinstance(eval_system_prompt, str):
            eval_system_prompt = Variable(eval_system_prompt, requires_grad=False, role_description="system prompt for the evaluation")
        self.eval_system_prompt = eval_system_prompt
        if ((engine is None) and (SingletonBackwardEngine().get_engine() is None)):
            raise Exception("No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
        elif engine is None:
            engine = SingletonBackwardEngine().get_engine()
        if isinstance(engine, str):
            engine = get_engine(engine)
        self.engine = engine
        self.llm_call = LLMCall(self.engine, self.eval_system_prompt)

    def forward(self, instance: Variable):
        """
        Calls the ResponseEvaluation object.

        :param instance: The instance variable.
        :type instance: Variable
        :return: The result of the evaluation
        """
        return self.llm_call(instance)

class MultiFieldEvaluation(Module):
    def __init__(
        self,
        evaluation_instruction: Variable,
        role_descriptions: List[str],
        engine: Union[EngineLM, str] = None,
        system_prompt: Variable = None,
    ):
        """A module to compare two variables using a language model.

        :param evaluation_instruction: Instruction to use as prefix for the comparison, specifying the nature of the comparison.
        :type evaluation_instruction: Variable
        :param engine: The language model to use for the comparison.
        :type engine: EngineLM
        :param v1_role_description: Role description for the first variable, defaults to "prediction to evaluate"
        :type v1_role_description: str, optional
        :param v2_role_description: Role description for the second variable, defaults to "correct result"
        :type v2_role_description: str, optional
        :param system_prompt: System prompt to use for the comparison, defaults to "You are an evaluation system that compares two variables."
        :type system_prompt: Variable, optional
        
        :example:
        TODO: Add an example
        """
        super().__init__()
        self.evaluation_instruction = evaluation_instruction
        if ((engine is None) and (SingletonBackwardEngine().get_engine() is None)):
            raise Exception("No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
        elif engine is None:
            engine = SingletonBackwardEngine().get_engine()
        if isinstance(engine, str):
            engine = get_engine(engine)
        self.engine = engine
        self.role_descriptions = role_descriptions
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = Variable("You are an evaluation system that compares two variables.",
                                            requires_grad=False,
                                            role_description="system prompt for the evaluation")
        format_string_items = ["{{instruction}}"]
        for role_description in role_descriptions:
            format_string_items.append(f"**{role_description}**: {{{role_description}}}")
        format_string = "\n".join(format_string_items)
        self.format_string = format_string.format(instruction=self.evaluation_instruction, **{role_description: "{"+role_description+"}" for role_description in role_descriptions})
        self.fields = {"instruction": self.evaluation_instruction, **{role_description: None for role_description in role_descriptions}}
        self.formatted_llm_call = FormattedLLMCall(engine=self.engine,
                                                   format_string=self.format_string,
                                                   fields=self.fields,
                                                   system_prompt=self.system_prompt)

    def forward(self, inputs: List[Variable]):
        for role_description, var in zip(self.role_descriptions, inputs):
            var.set_role_description(role_description)
        inputs_call = {"instruction": self.evaluation_instruction, 
                       **{role_description: var for role_description, var in zip(self.role_descriptions, inputs)}}
        return self.formatted_llm_call(inputs=inputs_call,
                                       response_role_description=f"evaluation of the a prediction")


class MultiFieldTokenParsedEvaluation(MultiFieldEvaluation):
    def __init__(
        self,
        evaluation_instruction: Variable,
        role_descriptions: List[str],
        engine: Union[EngineLM, str] = None,
        system_prompt: Variable = None,
        parse_tags: List[str] = None,
    ):
        super().__init__(
            evaluation_instruction=evaluation_instruction,
            role_descriptions=role_descriptions,
            engine=engine,
            system_prompt=system_prompt,
        )
        self.parse_tags = parse_tags

    def parse_output(self, response: Variable) -> str:
        """
        Parses the output response and returns the parsed response.

        :param response: The response to be parsed.
        :type response: Variable
        :return: The parsed response.
        :rtype: str
        """
        response_text = response.value
        response = response_text.split(self.parse_tags[0])[1].split(self.parse_tags[1])[0].strip()
        return response


DEFAULT_TEST_TIME = "You are an intelligent assistant used as an evaluator, and part of an optimization system. You will analyze a solution to a multi-choice problem. Investigate the reasoning and answer. Do not try to solve the problem, only raise the potential issues and mistakes in the answer. Be creative, think about different perspectives, and be very critical."

class MultiChoiceTestTime(Module):
    def __init__(self,
                 evaluation_instruction: str,
                 engine: Union[EngineLM, str] = None,
                 system_prompt: Variable = None):
        """
        The test-time loss to use when working on a response to a multiple choice question.

        :param evaluation_instruction: Instruction to guide the test time evaluation. This will be a prefix to the prompt.
        :type evaluation_instruction: str
        :param engine: LLM engine to use for the test-time loss computation.
        :type engine: EngineLM
        :param system_prompt: System prompt for the test-time loss computation, defaults to None
        :type system_prompt: Variable, optional
        """
        super().__init__()
        if system_prompt:
            self.tt_system_prompt = system_prompt
        else:
            tt_system_prompt = DEFAULT_TEST_TIME
            self.tt_system_prompt = Variable(tt_system_prompt,
                                                requires_grad=False,
                                                role_description="system prompt for the test-time evaluation")
        
        if ((engine is None) and (SingletonBackwardEngine().get_engine() is None)):
            raise Exception("No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
        elif engine is None:
            engine = SingletonBackwardEngine().get_engine()
        if isinstance(engine, str):
            engine = get_engine(engine)
        self.engine = engine
        format_string = "{instruction}\nQuestion: {{question}}\nAnswer by the language model: {{prediction}}"
        self.format_string = format_string.format(instruction=evaluation_instruction)
        self.fields = {"prediction": None, "question": None}
        self.formatted_llm_call = FormattedLLMCall(engine=self.engine,
                                                   format_string=self.format_string,
                                                   fields=self.fields,
                                                   system_prompt=self.tt_system_prompt)

    def forward(self, question: str, prediction: Variable) -> Variable:
        question_variable = Variable(question, 
                                     requires_grad=False, 
                                     role_description="the multiple choice question")

        inputs = {"prediction": prediction, "question": question_variable}
        return self.formatted_llm_call(inputs=inputs,
                                       response_role_description=f"evaluation of the {prediction.get_role_description()}")

class ImageQALoss(Module):
    def __init__(self,
                 evaluation_instruction: str,
                 engine: Union[EngineLM, str] = None,
                 system_prompt: Variable = None):
        super().__init__()
        self.evaluation_instruction = Variable(evaluation_instruction, role_description="evaluation instruction", requires_grad=False)
        if ((engine is None) and (SingletonBackwardEngine().get_engine() is None)):
            raise Exception("No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
        elif engine is None:
            engine = SingletonBackwardEngine().get_engine()
        if isinstance(engine, str):
            engine = get_engine(engine)
        self.engine = engine
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = Variable("You are an evaluation system that evaluates image-related questions.",
                                            requires_grad=False,
                                            role_description="system prompt for the evaluation")

        self.multimodal_llm_call = OrderedFieldsMultimodalLLMCall(engine=self.engine,
                                                                  system_prompt=self.system_prompt,
                                                                  fields=["Evaluation Instruction", "Question", "Image", "Answer"])

    def forward(self, image: Variable, question: Variable, response: Variable) -> Variable:
        
        inputs = {
            "Evaluation Instruction": self.evaluation_instruction,
            "Question": question,
            "Image": image,
            "Answer": response
        }
        return self.multimodal_llm_call(inputs=inputs,
                                        response_role_description=f"evaluation of the {response.get_role_description()}")