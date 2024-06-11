# This is how we save gradients to the variable.
GRADIENT_TEMPLATE = (
    "Here is a conversation:\n\n<CONVERSATION>{context}</CONVERSATION>\n\n"
    "This conversation is potentially part of a larger system. The output is used as {response_desc}\n\n"
    "Here is the feedback we got for {variable_desc} in the conversation:\n\n<FEEDBACK>{feedback}</FEEDBACK>\n\n"
)
