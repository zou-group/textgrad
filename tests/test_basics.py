import os
import pytest
from typing import Union, List
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

class DummyMultimodalEngine(EngineLM):

    def __init__(self, is_multimodal=False):
        self.is_multimodal = is_multimodal
        self.model_string = "gpt-4o" # fake

    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt: str = None, **kwargs):
        if isinstance(content, str):
            return "Hello Text"

        elif isinstance(content, list):
            has_multimodal_input = any(isinstance(item, bytes) for item in content)
            if (has_multimodal_input) and (not self.is_multimodal):
                raise NotImplementedError("Multimodal generation is only supported for Claude-3 and beyond.")

            return "Hello Text from Image"

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


def test_set_backward_engine():
    from textgrad.config import set_backward_engine, SingletonBackwardEngine
    from textgrad.engine.openai import ChatOpenAI
    from textgrad.engine_experimental.litellm import LiteLLMEngine

    engine = ChatOpenAI()
    set_backward_engine(engine, override=False)
    assert SingletonBackwardEngine().get_engine() == engine

    new_engine = LiteLLMEngine(model_string="gpt-3.5-turbo-0613")
    set_backward_engine(new_engine, True)
    assert SingletonBackwardEngine().get_engine() == new_engine

    with pytest.raises(Exception):
        set_backward_engine(engine, False)

def test_get_engine():
    from textgrad.engine import get_engine
    from textgrad.engine.openai import ChatOpenAI
    from textgrad.engine_experimental.litellm import LiteLLMEngine

    engine = get_engine("gpt-3.5-turbo-0613")
    assert isinstance(engine, ChatOpenAI)

    engine = get_engine("experimental:claude-3-opus-20240229")
    assert isinstance(engine, LiteLLMEngine)

    engine = get_engine("experimental:claude-3-opus-20240229", cache=True)
    assert isinstance(engine, LiteLLMEngine)

    engine = get_engine("experimental:claude-3-opus-20240229", cache=False)
    assert isinstance(engine, LiteLLMEngine)

    # get local diskcache
    from diskcache import Cache
    cache = Cache("./cache")

    engine = get_engine("experimental:claude-3-opus-20240229", cache=cache)
    assert isinstance(engine, LiteLLMEngine)

    with pytest.raises(ValueError):
        get_engine("invalid-engine")

    with pytest.raises(ValueError):
        get_engine("experimental:claude-3-opus-20240229", cache=[1,2,3])

    with pytest.raises(ValueError):
        get_engine("gpt-4o", cache=True)


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


def test_multimodal():
    from textgrad.autograd import MultimodalLLMCall, LLMCall
    from textgrad import Variable
    import httpx

    image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
    image_data = httpx.get(image_url).content

    os.environ['OPENAI_API_KEY'] = "fake_key"
    engine = DummyMultimodalEngine(is_multimodal=True)

    image_variable = Variable(image_data,
                              role_description="image to answer a question about", requires_grad=False)

    text = Variable("Hello", role_description="A variable")
    question_variable = Variable("What do you see in this image?", role_description="question", requires_grad=False)
    response = MultimodalLLMCall(engine=engine)([image_variable, question_variable])

    assert response.value == "Hello Text from Image"

    response = LLMCall(engine=engine)(text)

    assert response.value == "Hello Text"

    ## llm call cannot handle images
    with pytest.raises(AttributeError):
        response = LLMCall(engine=engine)([text, image_variable])

    # this is just to check the content, we can't really have double variables but
    # it's just for testing purposes

    with pytest.raises(AssertionError):
        response = MultimodalLLMCall(engine=engine)([Variable(4.2, role_description="tst"),
                                                 Variable(5.5, role_description="tst")])


def test_multimodal_from_url():
    from textgrad import Variable
    import httpx

    image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
    image_data = httpx.get(image_url).content

    image_variable = Variable(image_path=image_url,
                              role_description="image to answer a question about", requires_grad=False)

    image_variable_2 = Variable(image_data,
                                role_description="image to answer a question about", requires_grad=False)

    assert image_variable_2.value == image_variable.value