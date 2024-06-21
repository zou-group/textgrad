import os
import pytest
import logging


from textgrad import Variable, TextualGradientDescent, BlackboxLLM, sum
from textgrad.engine.base import EngineLM
from textgrad.engine.openai import ChatOpenAI
from textgrad.autograd import LLMCall, FormattedLLMCall

logging.disable(logging.CRITICAL)

# Dummy engine that always returns "Hello World"
class DummyEngine(EngineLM):
    def generate(self, prompt, system_prompt=None, **kwargs):
        return "Hello World"

    def __call__(self, prompt, system_prompt=None):
        return self.generate(prompt)

# Idempotent engine that returns the prompt as is
class IdempotentEngine(EngineLM):
    def generate(self, prompt, system_prompt=None, **kwargs):
        return prompt

    def __call__(self, prompt, system_prompt=None):
        return self.generate(prompt)

# Test if the variable object is passed correctly to the optimizer
def test_variable_object_passing():
    v = Variable("Hello", role_description="A variable")
    g = Variable("grad", role_description="A gradient")
    optimizer = TextualGradientDescent(parameters=[v], engine=DummyEngine())

    optimizer.parameters[0].gradients.add(g)
    optimizer.parameters[0].set_value("World")

    assert optimizer.parameters[0].get_value() == "World"
    assert optimizer.parameters[0].get_gradient_text() == "grad"
    assert v.value == "World"

# Test the OpenAI engine initialization
def test_openai_engine():
    with pytest.raises(ValueError):
        engine = ChatOpenAI()

    os.environ['OPENAI_API_KEY'] = "fake_key"
    engine = ChatOpenAI()

# Test importing main components from textgrad
def test_import_main_components():
    from textgrad import Variable, TextualGradientDescent, EngineLM
    from textgrad.engine.openai import ChatOpenAI
    from textgrad import BlackboxLLM

    assert Variable
    assert TextualGradientDescent
    assert EngineLM
    assert ChatOpenAI
    assert BlackboxLLM

# Test a simple forward pass using the dummy engine
def test_simple_forward_pass_engine():
    text = Variable("Hello", role_description="A variable")
    dummy_engine = DummyEngine()

    engine = BlackboxLLM(engine=dummy_engine)
    response = engine(text)

    assert response

# Test variable creation and attributes
def test_variable_creation():
    text = Variable("Hello", role_description="A variable")

    assert text.value == "Hello"
    assert text.role_description == "A variable"

    text = Variable("Hello", role_description="A variable", requires_grad=True)

    assert text.requires_grad is True
    assert text.value == "Hello"
    assert text.role_description == "A variable"

    with pytest.raises(TypeError):
        text = Variable("Hello")

# Test the sum function for variables
def test_sum_function():
    var1 = Variable("Line1", role_description="role1")
    var2 = Variable("Line2", role_description="role2")
    total = sum(variables=[var1, var2])

    assert total.get_value() == "Line1\nLine2"
    assert "a combination of the following" in total.get_role_description()
    assert "role1" in total.get_role_description()
    assert "role2" in total.get_role_description()

# Test LLMCall using the dummy engine
def test_llmcall():
    llm_call = LLMCall(DummyEngine())
    input_variable = Variable("Input", role_description="Input")
    output = llm_call(input_variable)

    assert isinstance(output, Variable)
    assert output.get_value() == "Hello World"
    assert input_variable in output.predecessors

# Test FormattedLLMCall using the idempotent engine
def test_formattedllmcall():
    format_string = "Question: {question}\nPrediction: {prediction}"
    fields = {"prediction": None, "question": None}
    formatted_llm_call = FormattedLLMCall(engine=IdempotentEngine(),
                                          format_string=format_string,
                                          fields=fields)
    inputs = {
        "question": Variable("q", role_description="Question"),
        "prediction": Variable("p", role_description="Prediction")
    }
    output = formatted_llm_call(inputs, response_role_description="test response")

    assert isinstance(output, Variable)
    assert output.get_value() == "Question: q\nPrediction: p"
    assert inputs["question"] in output.predecessors
    assert inputs["prediction"] in output.predecessors
    assert output.get_role_description() == "test response"
