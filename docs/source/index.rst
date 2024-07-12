.. TextGrad documentation master file, created by
   sphinx-quickstart on Sat May  4 17:35:54 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root toctree directive.
Welcome to TextGrad's documentation!
====================================

TextGrad is a Python package that provides a simple interface to implement LLM-"gradients" pipelines for text optimization!

Check out the :doc:`usage` section for further information, including how to
:doc:`install <installation>` the project. Want to directly jump to the optimization process?
Check out the :doc:`quickstart` guide!


.. image:: assets/logo_full.png
   :align: center

An autograd engine -- for textual gradients! 

TextGrad is a powerful framework building automatic ``differentiation'' via text.
TextGrad implements backpropagation through text feedback provided by LLMs, strongly building on the gradient metaphor.

We provide a simple and intuitive API that allows you to define your own loss functions and optimize them using text feedback.
This API is similar to the Pytorch API, making it simple to adapt to your use cases.

.. image:: assets/analogy.png
   :align: center

QuickStart
==========

If you know PyTorch, you know 80% of TextGrad.
Let's walk through the key components with a simple example. Say we want to use GPT-4o to generate a punchline for TextGrad.

.. code-block:: python

    import textgrad as tg
    # Step 1: Get an initial response from an LLM
    model = tg.BlackboxLLM("gpt-4o")
    punchline = model(tg.Variable("write a punchline for my github package about optimizing compound AI systems", role_description="prompt", requires_grad=False))
    punchline.set_role_description("a concise punchline that must hook everyone")

Initial `punchline` from the model:
> Supercharge your AI synergy with our optimization toolkit – where compound intelligence meets peak performance!

Not bad, but we (gpt-4o, I guess) can do better! Let's optimize the punchline using TextGrad.

.. code-block:: python

    # Step 2: Define the loss function and the optimizer, just like in PyTorch!
    loss_fn = tg.TextLoss("We want to have a super smart and funny punchline. Is the current one concise and addictive? Is the punch fun, makes sense, and subtle enough?")
    optimizer = tg.TGD(parameters=[punchline])

.. code-block:: python

    # Step 3: Do the loss computation, backward pass, and update the punchline
    loss = loss_fn(punchline)
    loss.backward()
    optimizer.step()

Optimized punchline:
> Boost your AI with our toolkit – because even robots need a tune-up!

Okay this model isn’t really ready for a comedy show yet (and maybe a bit cringy) but it is clearly trying. But who gets to maxima in one step? 

We have many more examples around how TextGrad can optimize all kinds of variables -- code, solutions to problems, molecules, prompts, and all that!

Tutorials
---------

We have prepared a couple of tutorials to get you started with TextGrad. 
You can run them directly in Google Colab by clicking on the links below.

.. |primiti| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Tutorial-Primitives.ipynb
    :alt: Open In Colab

.. |code| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/zou-group/textgrad/blob/main/examples/notebooks/Tutorial-Test-Time-Loss-for-Code.ipynb
    :alt: Open In Colab

.. |promptopt| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Tutorial-Prompt-Optimization.ipynb
    :alt: Open In Colab

.. |solut| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Tutorial-Solution-Optimization.ipynb
    :alt: Open In Colab



+--------------------------------------------------------------------------------+------------------+
| Name                                                                           | Link             |
+================================================================================+==================+
| Introduction to TextGrad primitives                                            | |primiti|        |
+--------------------------------------------------------------------------------+------------------+
| Code Optimization and New Loss Implementation                                  | |code|           |
+--------------------------------------------------------------------------------+------------------+
| Prompt Optimization                                                            | |promptopt|      |
+--------------------------------------------------------------------------------+------------------+
| Solution Optimization                                                          | |solut|          |
+--------------------------------------------------------------------------------+------------------+

Installation
============

You can install TextGrad via pip:

.. code-block:: bash

    pip install textgrad

Examples
========

Minimal Instance Optimization Example
-------------------------------------

TextGrad can optimize unstructured variables, such as text. Let us have an initial solution to a math problem that we want to improve. Here's how to do it with TextGrad, using GPT-4o:

.. code-block:: python

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

Output:

    Variable(value=Errors:
    1. Incorrect sign in the discriminant calculation: it should be b^2 - 4ac, not b^2 + 4ac.
    2. Incorrect simplification of the quadratic formula: the denominator should be 2a, not 6.
    3. Final solutions are missing the division by 2a., role=response from the language model, grads=)

.. code-block:: python

    loss.backward()
    optimizer.step()
    print(solution.value)

Output:

    To solve the equation 3x^2 - 7x + 2 = 0, we use the quadratic formula:
    x = (-b ± √(b^2 - 4ac)) / 2a

    Given:
    a = 3, b = -7, c = 2

    Substitute the values into the formula:
    x = (7 ± √((-7)^2 - 4(3)(2))) / 6
    x = (7 ± √(49 - 24)) / 6
    x = (7 ± √25) / 6
    x = (7 ± 5) / 6

    The solutions are:
    x1 = (7 + 5) / 6 = 12 / 6 = 2
    x2 = (7 - 5) / 6 = 2 / 6 = 1/3

Minimal Prompt Optimization Example
-----------------------------------

TextGrad can also optimize prompts in PyTorch style! Here's how to do it with TextGrad, using GPT-4o for feedback, and optimizing a prompt for gpt-3.5-turbo:

.. code-block:: python

    import textgrad as tg
    llm_engine = tg.get_engine("gpt-3.5-turbo")
    tg.set_backward_engine(tg.get_engine("gpt-4o"))

    _, val_set, _, eval_fn = load_task("BBH_object_counting", llm_engine)
    question_str, answer_str = val_set[0]
    question = tg.Variable(question_str, role_description="question to the LLM", requires_grad=False)
    answer = tg.Variable(answer_str, role_description="answer to the question", requires_grad=False)

Question:

    I have two stalks of celery, two garlics, a potato, three heads of broccoli, a carrot, and a yam. How many vegetables do I have?

Ground Truth Answer:

    10

.. code-block:: python

    system_prompt = tg.Variable("You are a concise LLM. Think step by step.",
                                requires_grad=True,
                                role_description="system prompt to guide the LLM's reasoning strategy for accurate responses")

    model = tg.BlackboxLLM(llm_engine, system_prompt=system_prompt)
    optimizer = tg.TGD(parameters=list(model.parameters()))

    prediction = model(question)

Prediction:

    You have a total of seven vegetables: two stalks of celery, two garlics, one potato, three heads of broccoli, one carrot, and one yam.

.. code-block:: python

    loss = eval_fn(inputs=dict(prediction=prediction, ground_truth_answer=answer))

Loss denoting accuracy:

    Variable(value=0, grads=)

.. code-block:: python

    loss.backward()

System prompt gradients:

    ... 2. **Encourage Explicit Summation**: - The prompt should encourage the model to explicitly state the summation process. This can help in verifying the accuracy of the count. For example, "Explain your calculations clearly and verify the total."....

.. code-block:: python

    optimizer.step()

New system prompt value:

    You are a concise LLM. Think step by step. Prioritize accuracy in your calculations. Identify and count each item individually. Explain your calculations clearly and verify the total. After calculating, review your steps to ensure the total is correct. If you notice a discrepancy in your count, re-evaluate the list and correct any mistakes.

.. code-block:: python

    prediction = model(question)

New prediction:

    Let's count the number of each vegetable:
    1. Celery stalks: 2
    2. Garlics: 2
    3. Potato: 1
    4. Broccoli heads: 3
    5. Carrot: 1
    6. Yam: 1

    Now, let's add up the total number of vegetables:
    2 + 2 + 1 + 3 + 1 + 1 = 10

    You have a total of 10 vegetables.

Resources
=========

Inspiration
-----------

Many existing works greatly inspired this project! Here is a non-exhaustive list:

- 📚 `PyTorch <https://github.com/pytorch/pytorch/>`_ The one and only. We owe a ton to PyTorch, hard to do justice here.
- 📚 `DSPy <https://github.com/stanfordnlp/dspy>`_ is a pioneer in writing LM-based programs in many different ways! Has been a huge inspiration for us.
- 📚 `Micrograd <https://github.com/karpathy/micrograd>`_: A tiny autograd engine greatly inspired our simple design!
- 📚 `ProTeGi <https://github.com/microsoft/LMOps/tree/main/prompt_optimization>`_: We owe the term "Textual Gradients" to ProTeGi!
- 📚 `Reflexion <https://github.com/noahshinn/reflexion>`_: A self-reflection that showed us the power of text-based reflection!

Citation
========

.. code-block:: bibtex

    @article{yuksekgonul2024textgrad,
      title={{TextGrad: Automatic ``Differentiation'' with Text}},
      author={Mert Yuksekgonul and Federico Bianchi and Joseph Boen and Sheng Liu and Zhi Huang and Carlos Guestrin and James Zou},
      year={2024},
    }

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   textgrad
   quickstart

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`