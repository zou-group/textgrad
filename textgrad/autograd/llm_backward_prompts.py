GLOSSARY_TEXT_BACKWARD = """
### Glossary of tags that will be sent to you:
# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model.
# - <OBJECTIVE_FUNCTION>: The objective of the optimization task.
# - <VARIABLE>: Specifies the span of the variable.
# - <ROLE>: The role description of the variable."""

### Backward engine prompts

# System prompt to the backward engine.
BACKWARD_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves a given text (i.e. the variable). You are the gradient (feedback) engine. "
    "Your only responsibility is to give intelligent and creative feedback and constructive criticism to variables, given an objective specified in <OBJECTIVE_FUNCTION> </OBJECTIVE_FUNCTION> tags. "
    "The variables may be solutions to problems, prompts to language models, code, or any other text-based variable. "
    "Pay attention to the role description of the variable, and the context in which it is used. You should assume that the variable will be used in a similar context in the future. "
    "Only provide strategies, explanations, and methods to change in the variable. DO NOT propose a new version of the variable, that will be the job of the optimizer. Your only job is to send feedback and criticism (compute 'gradients'). "
    "For instance, feedback can be in the form of 'Since language models have the X failure mode...', 'Adding X can fix this error because...', 'Removing X can improve the objective function because...', 'Changing X to Y would fix the mistake ...', that gets at the downstream objective.\n"
    "If a variable is already working well (e.g. the objective function is perfect, an evaluation shows the response is accurate), you should not give feedback.\n"
    f"{GLOSSARY_TEXT_BACKWARD}")

# First part of the prompt for the llm backward function
CONVERSATION_TEMPLATE = (
    "<LM_SYSTEM_PROMPT> {system_prompt} </LM_SYSTEM_PROMPT>\n\n"
    "<LM_INPUT> {prompt} </LM_INPUT>\n\n"
    "<LM_OUTPUT> {response_value} </LM_OUTPUT>\n\n"
)

# Has the gradient on the output.
CONVERSATION_START_INSTRUCTION_CHAIN = (
    "You will give feedback to a variable with the following role: <ROLE> {variable_desc} </ROLE>. "
    "Here is a conversation with a language model (LM):\n\n"
    "{conversation}"
)
OBJECTIVE_INSTRUCTION_CHAIN = (
    "This conversation is part of a larger system. The <LM_OUTPUT> was later used as {response_desc}.\n\n"
    "<OBJECTIVE_FUNCTION>Your goal is to give feedback to the variable to address the following feedback on the LM_OUTPUT: {response_gradient} </OBJECTIVE_FUNCTION>\n\n"
)

# Does not have gradient on the output
CONVERSATION_START_INSTRUCTION_BASE = (
    "You will give feedback to a variable with the following role: <ROLE> {variable_desc} </ROLE>. "
    "Here is an evaluation of the variable using a language model:\n\n"
    "{conversation}"
)

OBJECTIVE_INSTRUCTION_BASE = (
    "<OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism to the variable given the above evaluation output. "
    "Our only goal is to improve the above metric, and nothing else. </OBJECTIVE_FUNCTION>\n\n"
)

# Third part of the prompt for the llm backward function.
# Asks the user to evaluate a variable in the conversation.
EVALUATE_VARIABLE_INSTRUCTION = (
    "We are interested in giving feedback to the {variable_desc} "
    "for this conversation. Specifically, give feedback to the following span "
    "of text:\n\n<VARIABLE> "
    "{variable_short} </VARIABLE>\n\n"
    "Given the above history, describe how the {variable_desc} "
    "could be improved to improve the <OBJECTIVE_FUNCTION>. Be very creative, critical, and intelligent.\n\n"
)

SEARCH_QUERY_BACKWARD_INSTRUCTION = (
    "Here is a query and a response from searching with {engine_name}:\n"
    "<QUERY> {query} </QUERY>\n"
    "<RESULTS> {results} </RESULTS>\n\n"
)


GRADIENT_OF_RESULTS_INSTRUCTION = (
    "For the search results from {engine_name} we got the following feedback:\n\n"
    "<FEEDBACK>{results_gradient}</FEEDBACK>\n\n"
)

IN_CONTEXT_EXAMPLE_PROMPT_ADDITION = (
    "You must base on the following examples when give feedback and criticism to the variable:\n\n"
    "<EXAMPLES>{in_context_examples}</EXAMPLES>\n\n"
)
