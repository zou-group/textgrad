import textgrad as tg
import vertexai

vertexai.init(project="your-project-id")

tg.set_backward_engine("vertex-gemini-1.5-flash-001", override=True)

# Step 1: Get an initial response from an LLM.
model = tg.BlackboxLLM("vertex-gemini-1.5-flash-001")
question_string = ("If it takes 1 hour to dry 25 shirts under the sun, "
                   "how long will it take to dry 30 shirts under the sun? "
                   "Reason step by step")

question = tg.Variable(question_string,
                       role_description="question to the LLM",
                       requires_grad=False)

answer = model(question)
