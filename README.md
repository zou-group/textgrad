![Logo](assets/logo_full.png)

<!--- BADGES: START --->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Prompt-Optimization.ipynb)
[![GitHub license](https://img.shields.io/badge/License-MIT-blue.svg)][#license-gh-package]
[![Arxiv](https://img.shields.io/badge/arXiv-2406.07496-B31B1B.svg)][#arxiv-paper-package]
[![Documentation Status](https://readthedocs.org/projects/textgrad/badge/?version=latest)][#docs-package]
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/textgrad)][#pypi-package]
[![PyPI](https://img.shields.io/pypi/v/textgrad)][#pypi-package]
[![Conda - Platform](https://img.shields.io/conda/pn/conda-forge/textgrad?logo=anaconda&style=flat)][#conda-forge-package]
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/textgrad?logo=anaconda&style=flat&color=orange)][#conda-forge-package]

[#license-gh-package]: https://lbesson.mit-license.org/
[#arxiv-paper-package]: https://arxiv.org/abs/2406.07496
[#docs-package]: https://textgrad.readthedocs.io/en/latest/?badge=latest
[#pypi-package]: https://pypi.org/project/textgrad/
[#conda-forge-package]: https://anaconda.org/conda-forge/textgrad
<!--- BADGES: END --->

## TextGrad: Automatic ''Differentiation'' via Text

An autograd engine -- for textual gradients!

TextGrad is a powerful framework  building automatic ``differentiation'' via text.
TextGrad implements backpropagation through text feedback provided by LLMs, strongly building on the gradient metaphor

We provide a simple and intuitive API that allows you to define your own loss functions and optimize them using text feedback.
This API is similar to the Pytorch API, making it simple to adapt to your usecases.

![Analogy with Torch](assets/analogy.png)


### Updates:

**29th Sept 2024**:

We are introducing a new engine based on [litellm](https://github.com/BerriAI/litellm). This should allow
you to use any model you like, as long as it is supported by litellm. This means that now
**Bedrock, Together, Gemini and even more** are all supported by TextGrad!

In addition to this, with the new engines it should be easy to enable and disable caching.  

We are in the process of testing these new engines and deprecating the old engines. If you have any issues, please let us know!

The new litellm engines can be loaded with the following code: 

An example of loading a litellm engine:
```python
engine = get_engine("experimental:gpt-4o", cache=False)

# this also works with

set_backward_engine("experimental:gpt-4o", cache=False)
```

Be sure to set the relevant environment variables for the new engines!

An example of forward pass:
```python

import httpx
from textgrad.engine_experimental.litellm import LiteLLMEngine

LiteLLMEngine("gpt-4o", cache=True).generate(content="hello, what's 3+4", system_prompt="you are an assistant")

image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
image_data = httpx.get(image_url).content
```

In the examples folder you will find two new notebooks that show how to use the new engines.


## QuickStart
If you know PyTorch, you know 80% of TextGrad.
Let's walk through the key components with a simple example. Say we want to use GPT-4o to solve a simple
reasoning problem.

The question is *If it takes 1 hour to dry 25 shirts under the sun, how long will it take to dry 30 shirts under the sun? Reason step by step.* (Thanks, [Reddit User](https://www.reddit.com/r/OpenAI/comments/18q479x/comment/kf444es/))

```python
import textgrad as tg

tg.set_backward_engine("gpt-4o", override=True)

# Step 1: Get an initial response from an LLM.
model = tg.BlackboxLLM("gpt-4o")
question_string = ("If it takes 1 hour to dry 25 shirts under the sun, "
                   "how long will it take to dry 30 shirts under the sun? "
                   "Reason step by step")

question = tg.Variable(question_string,
                       role_description="question to the LLM",
                       requires_grad=False)

answer = model(question)
```

> :warning: **answer: To determine how long it will take to dry 30 shirts under the sun,**
> **we can use a proportional relationship based on the given information.**
> **Here’s the step-by-step reasoning: [.....]**
> **So, it will take 1.2 hours (or 1 hour and 12 minutes) to dry 30 shirts under the sun.**


As you can see, **the model's answer is incorrect.** We can optimize the answer using TextGrad to get the correct answer.

```python

answer.set_role_description("concise and accurate answer to the question")

# Step 2: Define the loss function and the optimizer, just like in PyTorch!
# Here, we don't have SGD, but we have TGD (Textual Gradient Descent)
# that works with "textual gradients".
optimizer = tg.TGD(parameters=[answer])
evaluation_instruction = (f"Here's a question: {question_string}. "
                           "Evaluate any given answer to this question, "
                           "be smart, logical, and very critical. "
                           "Just provide concise feedback.")


# TextLoss is a natural-language specified loss function that describes
# how we want to evaluate the reasoning.
loss_fn = tg.TextLoss(evaluation_instruction)
```
> :brain: **loss: [...] Your step-by-step reasoning is clear and logical,**
> **but it contains a critical flaw in the assumption that drying time is**
> **directly proportional to the number of shirts. [...]**

```python
# Step 3: Do the loss computation, backward pass, and update the punchline.
# Exact same syntax as PyTorch!
loss = loss_fn(answer)
loss.backward()
optimizer.step()
answer
```

> :white_check_mark: **answer: It will still take 1 hour to dry 30 shirts under the sun,**
> **assuming they are all laid out properly to receive equal sunlight.**

### Updates:

**29th Sept 2024**:

We are introducing a new engine based on [litellm](https://github.com/BerriAI/litellm). This should allow
you to use any model you like, as long as it is supported by litellm. This means that now
**Bedrock, Together, Gemini and even more** are all supported by TextGrad!

In addition to this, with the new engines it should be easy to enable and disable caching.  

We are in the process of testing these new engines and deprecating the old engines. If you have any issues, please let us know!

The new litellm engines can be loaded with the following code: 

An example of loading a litellm engine:
```python
engine = get_engine("experimental:gpt-4o", cache=False)

# this also works with

set_backward_engine("experimental:gpt-4o", cache=False)
```

An example of forward pass:
```python

import httpx
from textgrad.engine_experimental.litellm import LiteLLMEngine

LiteLLMEngine("gpt-4o", cache=True).generate(content="hello, what's 3+4", system_prompt="you are an assistant")

image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
image_data = httpx.get(image_url).content
```

In the examples folder you will find two new notebooks that show how to use the new engines.


We have many more examples around how TextGrad can optimize all kinds of variables -- code, solutions to problems, molecules, prompts, and all that!

### Tutorials

We have prepared a couple of tutorials to get you started with TextGrad. The order of this
tutorial is what we would recommend to follow for a beginner. You can run them directly in Google Colab by clicking on the links below (but
you need an OpenAI/Anthropic key to run the LLMs).

<div align="center">

| Tutorial                                           | Difficulty                                                      | Colab Link                                                                                                                                                                                                    |
|----------------------------------------------------|-----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1. Introduction to TextGrad Primitives             | ![](https://img.shields.io/badge/Level-Beginner-green.svg)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Tutorial-Primitives.ipynb)              |
| 2. Solution Optimization                           | ![](https://img.shields.io/badge/Level-Beginner-green.svg)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Tutorial-Solution-Optimization.ipynb)   |
| 3. Optimizing a Code Snippet and Define a New Loss | ![](https://img.shields.io/badge/Level-Beginner-green.svg)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/textgrad/blob/main/examples/notebooks/Tutorial-Test-Time-Loss-for-Code.ipynb) |
| 4. Prompt Optimization                             | ![](https://img.shields.io/badge/Level-Intermediate-yellow.svg) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Tutorial-Prompt-Optimization.ipynb)     |
| 5. MultiModal Optimization                         | ![](https://img.shields.io/badge/Level-Beginner-green.svg)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Tutorial-MultiModal.ipynb)              |

</div>



### Installation

You can install TextGrad using any of the following methods.

**With `pip`**:

```bash
pip install textgrad
```

**With `conda`**:

```sh
conda install -c conda-forge textgrad
```

> :bulb: The conda-forge package for `textgrad` is maintained [here](https://github.com/conda-forge/textgrad-feedstock).

**Bleeding edge installation with `pip`**:

```sh
pip install git+https://github.com/zou-group/textgrad.git
```

**Installing textgrad with vllm**:

```sh
pip install textgrad[vllm]
```

See [here](https://pip.pypa.io/en/stable/cli/pip_install/) for more details on various methods of pip installation.

## More detailed examples

### Minimal Instance Optimization Example

TextGrad can optimize unstructured variables, such as text. Let us have an initial solution to a math problem that we want to improve. Here's how to do it with TextGrad, using GPT-4o:

```python
tg.set_backward_engine("gpt-4o")

initial_solution = """To solve the equation 3x^2 - 7x + 2 = 0, we use the quadratic formula:
x = (-b ± √(b^2 - 4ac)) / 2a
a = 3, b = -7, c = 2
x = (7 ± √((-7)^2 - 4 * 3(2))) / 6
x = (7 ± √(7^3) / 6
The solutions are:
x1 = (7 + √73)
x2 = (7 - √73)"""

# Define the variable to optimize, let requires_grad=True to enable gradient computation
solution = tg.Variable(initial_solution,
                       requires_grad=True,
                       role_description="solution to the math question")

# Define the optimizer, let the optimizer know which variables to optimize, and run the loss function

loss_fn = tg.TextLoss("You will evaluate a solution to a math question. Do not attempt to solve it yourself, do not give a solution, only identify errors. Be super concise.")

optimizer = tg.TGD(parameters=[solution])
loss = loss_fn(solution)
```

Output:
> Variable(value=Errors:
> 1. Incorrect sign in the discriminant calculation: it should be b^2 - 4ac, not b^2 + 4ac.
> 2. Incorrect simplification of the quadratic formula: the denominator should be 2a, not 6.
> 3. Final solutions are missing the division by 2a., role=response from the language model, grads=)

```python
loss.backward()
optimizer.step()
print(solution.value)
```

Output:
> To solve the equation 3x^2 - 7x + 2 = 0, we use the quadratic formula:
> x = (-b ± √(b^2 - 4ac)) / 2a
>
> Given:
> a = 3, b = -7, c = 2
>
> Substitute the values into the formula:
> x = (7 ± √((-7)^2 - 4(3)(2))) / 6
> x = (7 ± √(49 - 24)) / 6
> x = (7 ± √25) / 6
> x = (7 ± 5) / 6
>
> The solutions are:
> x1 = (7 + 5) / 6 = 12 / 6 = 2
> x2 = (7 - 5) / 6 = 2 / 6 = 1/3

### Minimal Prompt Optimization Example
TextGrad can also optimize prompts in PyTorch style! Here's how to do it with TextGrad, using GPT-4o for feedback, and optimizing a prompt for gpt-3.5-turbo:
```python
import textgrad as tg
from textgrad.tasks import load_task

llm_engine = tg.get_engine("gpt-3.5-turbo")
tg.set_backward_engine("gpt-4o")

_, val_set, _, eval_fn = load_task("BBH_object_counting", llm_engine)
question_str, answer_str = val_set[0]
question = tg.Variable(question_str, role_description="question to the LLM", requires_grad=False)
answer = tg.Variable(answer_str, role_description="answer to the question", requires_grad=False)
```

Question:
> I have two stalks of celery, two garlics, a potato, three heads of broccoli, a carrot, and a yam. How many vegetables do I have?

Ground Truth Answer:
> 10

```python
system_prompt = tg.Variable("You are a concise LLM. Think step by step.",
                            requires_grad=True,
                            role_description="system prompt to guide the LLM's reasoning strategy for accurate responses")

model = tg.BlackboxLLM(llm_engine, system_prompt=system_prompt)
optimizer = tg.TGD(parameters=list(model.parameters()))

prediction = model(question)
```

Prediction:
> You have a total of seven vegetables: two stalks of celery, two garlics, one potato, three heads of broccoli, one carrot, and one yam.

```python
loss = eval_fn(inputs=dict(prediction=prediction, ground_truth_answer=answer))
```

Loss denoting accuracy:
> Variable(value=0, grads=)

```python
loss.backward()
```

System prompt gradients:
> ... 2. **Encourage Explicit Summation**: - The prompt should encourage the model to explicitly state the summation process. This can help in verifying the accuracy of the count. For example, "Explain your calculations clearly and verify the total."....

```python
optimizer.step()
```

New system prompt value:
> You are a concise LLM. Think step by step. Prioritize accuracy in your calculations. Identify and count each item individually. Explain your calculations clearly and verify the total. After calculating, review your steps to ensure the total is correct. If you notice a discrepancy in your count, re-evaluate the list and correct any mistakes.

```python
prediction = model(question)
```

New prediction:
> Let's count the number of each vegetable:
> 1. Celery stalks: 2
> 2. Garlics: 2
> 3. Potato: 1
> 4. Broccoli heads: 3
> 5. Carrot: 1
> 6. Yam: 1
>
> Now, let's add up the total number of vegetables:
> 2 + 2 + 1 + 3 + 1 + 1 = 10
>
> You have a total of 10 vegetables.

## Resources

### Inspiration
Many existing works greatly inspired this project! Here is a non-exhaustive list:
- 📚 [PyTorch](https://github.com/pytorch/pytorch/) The one and only. We owe a ton to PyTorch, hard to do justice here.
- 📚 [DSPy](https://github.com/stanfordnlp/dspy) is a pioneer in writing LM-based programs in many different ways! Has been a huge inspiration for us.
- 📚 [Micrograd](https://github.com/karpathy/micrograd): A tiny autograd engine greatly inspired our simple design!
- 📚 [ProTeGi](https://github.com/microsoft/LMOps/tree/main/prompt_optimization): We owe the term "Textual Gradients" to ProTeGi!
- 📚 [Reflexion](https://github.com/noahshinn/reflexion): A self-reflection that showed us the power of text-based reflection!

### Citation
```bibtex
@article{yuksekgonul2024textgrad,
      title={TextGrad: Automatic "Differentiation" via Text},
      author={Mert Yuksekgonul and Federico Bianchi and Joseph Boen and Sheng Liu and Zhi Huang and Carlos Guestrin and James Zou},
      year={2024},
      eprint={2406.07496},
      archivePrefix={arXiv}
}
```


### Contributors

We are grateful for all the help we got from our contributors!

<!-- readme: contributors -start -->
<table>
	<tbody>
		<tr>
            <td align="center">
                <a href="https://github.com/vinid">
                    <img src="https://avatars.githubusercontent.com/u/2234699?v=4" width="100;" alt="vinid"/>
                    <br />
                    <sub><b>Federico Bianchi</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/mertyg">
                    <img src="https://avatars.githubusercontent.com/u/29640736?v=4" width="100;" alt="mertyg"/>
                    <br />
                    <sub><b>Mert Yuksekgonul</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/nihalnayak">
                    <img src="https://avatars.githubusercontent.com/u/5679782?v=4" width="100;" alt="nihalnayak"/>
                    <br />
                    <sub><b>Nihal Nayak</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/sugatoray">
                    <img src="https://avatars.githubusercontent.com/u/10201242?v=4" width="100;" alt="sugatoray"/>
                    <br />
                    <sub><b>Sugato Ray</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/lupantech">
                    <img src="https://avatars.githubusercontent.com/u/17663606?v=4" width="100;" alt="lupantech"/>
                    <br />
                    <sub><b>Pan Lu</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/ruanwz">
                    <img src="https://avatars.githubusercontent.com/u/4874?v=4" width="100;" alt="ruanwz"/>
                    <br />
                    <sub><b>David Ruan</b></sub>
                </a>
            </td>
		</tr>
		<tr>
            <td align="center">
                <a href="https://github.com/sanowl">
                    <img src="https://avatars.githubusercontent.com/u/99511815?v=4" width="100;" alt="sanowl"/>
                    <br />
                    <sub><b>San </b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/huangzhii">
                    <img src="https://avatars.githubusercontent.com/u/10902358?v=4" width="100;" alt="huangzhii"/>
                    <br />
                    <sub><b>Zhi Huang</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/ajms">
                    <img src="https://avatars.githubusercontent.com/u/2394425?v=4" width="100;" alt="ajms"/>
                    <br />
                    <sub><b>Albert</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/tboen1">
                    <img src="https://avatars.githubusercontent.com/u/46495807?v=4" width="100;" alt="tboen1"/>
                    <br />
                    <sub><b>tboen1</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/AtakanTekparmak">
                    <img src="https://avatars.githubusercontent.com/u/59488384?v=4" width="100;" alt="AtakanTekparmak"/>
                    <br />
                    <sub><b>Atakan Tekparmak</b></sub>
                </a>
            </td>
		</tr>
	<tbody>
</table>
<!-- readme: contributors -end -->
