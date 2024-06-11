.. TextGrad documentation master file, created by
   sphinx-quickstart on Sat May  4 17:35:54 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TextGrad's documentation!
====================================

.. admonition:: TextGrad Gradients!
    :class: sidebar note

    TextGrad uses the gradient metaphor to describe the process of optimizing text. In this context, a "gradient" is
    not a mathematical gradient, but rather text feedback that suggest a "direction" in which to move the text to optimize it for a given task.

TextGrad is a Python package that provides a simple interface to implement LLM-"gradients" pipelines for text optimization!

Check out the :doc:`usage` section for further information, including how to
:ref:`install <installation>` the project. Want to directly jump to the optimization process?
Check out the :doc:`quickstart` guide!

TextGrad keeps track of variable in a way similar to torch's!

.. code-block:: python

    import textgrad as tg

    # Optimize text
    text_of_interest = "I am a text to optimize"
    var = tg.Variable(text_of_interest,
                      requires_grad=True)

    print(var)


Contents
--------

.. toctree::
   :maxdepth: 1

   usage
   textgrad
   quickstart

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Creating recipes
----------------
