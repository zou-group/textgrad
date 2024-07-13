import os
import logging
import pytest
from typing import Union, List

from textgrad import Variable, BlackboxLLM, TextLoss
from textgrad.optimizer import TextualGradientDescent
from textgrad.engine.vllm import ChatVLLM

logging.disable(logging.CRITICAL)
vllm_engine = ChatVLLM(model_string="meta-llama/Meta-Llama-3-8B-Instruct")

def test_import_vllm():
    assert ChatVLLM

def test_simple_forward_pass_engine():
    text = Variable("Hello", role_description="A variable")
    engine = BlackboxLLM(engine=vllm_engine)
    response = engine(text)

    assert response

def test_primitives():
    """
    Test the basic functionality of the Variable class.
    """
    x = Variable("A sntence with a typo", role_description="The input sentence", requires_grad=True)
    system_prompt = Variable("Evaluate the correctness of this sentence", role_description="The system prompt")
    loss = TextLoss(system_prompt, engine=vllm_engine)
    optimizer = TextualGradientDescent(parameters=[x], engine=vllm_engine)

    l = loss(x)
    l.backward(vllm_engine)
    optimizer.step()

    assert x.value == "A sentence with a typo"
