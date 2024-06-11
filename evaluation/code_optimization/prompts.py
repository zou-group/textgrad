CODE_INSTANCE_ROLE_DESCRIPTION = "Code generated that must be evaluated for correctness and runtime performance"
SYSTEM_PROMPT_FOR_FIRST_CODE = """You are an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature).
Use a Python code block to write your response. For example:
```python
print('Hello world!')
```"""

from textgrad.autograd.function import Module
from textgrad.autograd import FormattedLLMCall
from textgrad import EngineLM
from textgrad import Variable

DEFAULT_TEST_TIME_WITH_TESTS = """You are an intelligent assistant used as an evaluator, and part of an optimization system. 
You will analyze a code implementation for a coding problem and unit test results. 
The code will be tested with harder tests, so do not just check if the code passes the provided tests.
Think about the correctness of the code and its performance in harder test cases.
Give very concise feedback.
Investigate the code problem and the provided implementation. 
For each failed unit test case, start analyzing it by saying "The code did not pass this test because...". 
Explain why the current implementation is not getting the expected output. Do not provide a revised implementation. Carefully 
suggest why there are issues with the code and provide feedback.
"""

default_instruction_test = (f"")

class CodeTestTimewithTests(Module):
    def __init__(self,
                 engine: EngineLM,
                 evaluation_instruction: str = default_instruction_test,
                 system_prompt: Variable = None):
        super().__init__()
        if system_prompt:
            self.tt_system_prompt = system_prompt
        else:
            tt_system_prompt = DEFAULT_TEST_TIME_WITH_TESTS
            self.tt_system_prompt = Variable(tt_system_prompt,
                                                requires_grad=False,
                                                role_description="system prompt for the evaluation of the code solution")
        self.engine = engine
        format_string = "You are a language model that evaluates a python code snippet.\n"
        format_string += "{instruction}\n**The coding problem:**\n\n{{problem}}\n**{role}**{{program}}**\n\nThe test results:**\n\n{{tests}}\n"
        self.format_string = format_string.format(instruction=evaluation_instruction, role=CODE_INSTANCE_ROLE_DESCRIPTION)
        self.fields = {"problem": None, "program": None, "tests": None}
        self.formatted_llm_call = FormattedLLMCall(engine=self.engine,
                                                   format_string=self.format_string,
                                                   fields=self.fields,
                                                   system_prompt=self.tt_system_prompt)

    def forward(self, problem: str, program: Variable, tests: str) -> Variable:
        problem_variable = Variable(problem,
                                     requires_grad=False,
                                     role_description="the coding problem")
        tests = Variable(tests,
                         requires_grad=False,
                         role_description="the results of the unit tests on the code solution")
        inputs = {"program": program, "problem": problem_variable, "tests": tests}
        return self.formatted_llm_call(inputs=inputs,
                                       response_role_description=f"evaluation of the {program.get_role_description()}")
