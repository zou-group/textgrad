# TextGrad: Automatic ''Differentiation'' via Text

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Prompt-Optimization.ipynb)
[![GitHub license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Arxiv](http://img.shields.io/badge/arXiv-0000.00000-B31B1B.svg)](https://arxiv.org/abs/0000.00000)

![Logo](assets/logo_full.png)

An autograd engine -- for textual gradients! 

TextGrad is a powerful framework  building automatic ``differentiation'' via text.
TextGrad implements backpropagation through text feedback provided by LLMs, strongly building on the gradient metaphor

We provide a simple and intuitive API that allows you to define your own loss functions and optimize them using text feedback.
This API is similar to the Pytorch API, making it simple to adapt to your usecases.

![Analogy with Torch](assets/analogy.png)

## QuickStart

### Tutorials

We have prepared a couple of tutorials to get you started with TextGrad. 
You can run them directly in Google Colab by clicking on the links below.

<div align="center">

| Example                             | Colab Link                                                                                                                                                                                                    |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Introduction to TextGrad Primitives | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/)              |
| Prompt Optimization                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Prompt-Optimization.ipynb)     |
| Solution Optimization               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Tutorial-Solution-Optimization.ipynb)   |
| Optimizing a code snippet and defining a test time loss  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/textgrad/blob/main/examples/notebooks/Tutorial-Test-Time-Loss-for-Code.ipynb) |

</div>

### Installation

You can install TextGrad via pip:

```bash
pip install textgrad
```

## Examples

### Minimal Instance Optimization Example

TextGrad can optimize unstructured variables, such as text. Let us have an initial solution to a math problem that we want to improve. Here's how to do it with TextGrad, using GPT-4o:

```python
import textgrad as tg
tg.set_backward_engine(tg.get_engine("gpt-4o"))

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

# Define the loss function, via a system prompt to an LLM
loss_system_prompt = tg.Variable("""You will evaluate a solution to a math question. Do not attempt to solve it yourself, do not give a solution, only identify errors. Be super concise.""",
                                 requires_grad=False,
                                 role_description="system prompt")

loss_fn = tg.TextLoss(loss_system_prompt)

# Define the optimizer, let the optimizer know which variables to optimize
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
llm_engine = tg.get_engine("gpt-3.5-turbo")
tg.set_backward_engine(tg.get_engine("gpt-4o"))

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
  title={{TextGrad: Automatic ``Differentiation'' with Text}},
  author={Mert Yuksekgonul and Federico Bianchi and Joseph Boen and Sheng Liu and Zhi Huang and Carlos Guestrin and James Zou},
  year={2024},
}
``` 