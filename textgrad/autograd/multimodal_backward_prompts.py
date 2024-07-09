# First part of the prompt for the llm backward function
MULTIMODAL_CONVERSATION_TEMPLATE = (
    "\n Above messages are the <LM_INPUT>\n\n"
    "<LM_SYSTEM_PROMPT> {system_prompt} </LM_SYSTEM_PROMPT>\n\n"
    "<LM_OUTPUT> {response_value} </LM_OUTPUT>\n\n"
)
