QuickStart
==========

What can TextGrad do? TextGrad can optimize your prompts from a language model in an automatic way.


.. code-block:: python

    import textgrad

    # Set the backward engine as an External LLM API object.
    See textgrad.config for more details.
    textgrad.set_backward_engine(llm_api)
    basic_system_prompt = "You are a language model that summarizes \
                            a given document"

    system_prompt = textgrad.Variable(basic_system_prompt, requires_grad=True)

    api_model = textgrad.model.BlackboxLLM(llm_api)

    # this tells the model to use the following system prompt
    api_model = api_model + system_prompt

    big_document = "This is a big document that we want to summarize."

    # Since we will not need the criticisms for the document,
    # we will explicitly set requires_grad=False
    doc = textgrad.Variable(data, requires_grad=False)
    # Get the summary
    summary = api_model(big_document)

    # Compute a loss
    evaluation_prompt = "Evaluate if this is a good summary \
                        based on completeness and fluency."

    loss_fn = textgrad.ResponseEvaluation(engine=llm_api,
                    evaluation_instruction=Variable(evaluation_prompt,
                    requires_grad=False))

    loss = loss_fn(summary)
    loss.backward() # This populates gradients

    optimizer = textgrad.TextualGradientDescent(engine=llm_api,
    parameters=[system_prompt])
    optimizer.step()
    print(system_prompt)