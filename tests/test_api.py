import pytest
import logging
import random
import textgrad as tg
# We'll use below utilities to run a python function.
from IPython.core.interactiveshell import InteractiveShell

logging.disable(logging.CRITICAL)

SYSTEM_PROMPT = "You are a smart language model that evaluates code snippets. You do not solve problems or propose new code snippets, only evaluate existing solutions critically and give very concise feedback."
INSTRUCTION = """Think about the problem and the code snippet. Does the code solve the problem? What is the runtime complexity?"""

PROBLEM_TEXT = """"Longest Increasing Subsequence (LIS)
    
    Problem Statement:
    Given a sequence of integers, find the length of the longest subsequence that is strictly increasing. A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.
    
    Input:
    The input consists of a list of integers representing the sequence.
    
    Output:
    The output should be an integer representing the length of the longest increasing subsequence."""

INITIAL_SOLUTION = """
    def longest_increasing_subsequence(nums):
        n = len(nums)
        dp = [1] * n
    
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
    
        max_length = max(dp)
        lis = []
    
        for i in range(n - 1, -1, -1):
            if dp[i] == max_length:
                lis.append(nums[i])
                max_length -= 1
    
        return len(lis[::-1]) 
        """

BUGGED_SOLUTION = """
def longest_increasing_subsequence(nums):
        n = len(nums)
        dp = [1] * n
    
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
    
        max_length = max(dp)
        lis = []
    
        for i in range(n - 1, -1, -1):
            if dp[i] == max_length:
                lis.append(nums[i])
                max_length -= 1
    
        return len(lis[::-1])+1
        """


def generate_random_test_case(size, min_value, max_value):
    return [random.randint(min_value, max_value) for _ in range(size)]


def run_function_in_interpreter(func_code):
    interpreter = InteractiveShell.instance()

    interpreter.run_cell(func_code, store_history=False, silent=True)

    func_name = func_code.split("def ")[1].split("(")[0].strip()
    func = interpreter.user_ns[func_name]

    return func


def eval_function_with_asserts(fn):
    nums = [10, 22, 9, 33, 21, 50, 41, 60]
    assert fn(nums) == 5

    nums = [7, 2, 1, 3, 8, 4, 9, 6, 5]
    assert fn(nums) == 4

    nums = [5, 4, 3, 2, 1]
    assert fn(nums) == 1

    nums = [1, 2, 3, 4, 5]
    assert fn(nums) == 5

    nums = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    assert fn(nums) == 4

    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    assert fn(nums) == 4

    nums = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
    assert fn(nums) == 6

    nums = [7, 7, 7, 7, 7, 7, 7]
    assert fn(nums) == 1

    nums = [20, 25, 47, 35, 56, 68, 98, 101, 212, 301, 415, 500]
    assert fn(nums) == 11

    nums = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert fn(nums) == 1

    print("All test cases passed!")

def test_api():

    longest_increasing_subsequence = run_function_in_interpreter(INITIAL_SOLUTION)

    eval_function_with_asserts(longest_increasing_subsequence)

    llm_engine = tg.get_engine("experimental:gpt-4o-mini")
    tg.set_backward_engine(llm_engine)

    code = tg.Variable(value=INITIAL_SOLUTION,
                       requires_grad=True,
                       role_description="code instance to optimize")

    problem = tg.Variable(PROBLEM_TEXT,
                          requires_grad=False,
                          role_description="the coding problem")

    optimizer = tg.TGD(parameters=[code])

    loss_system_prompt = SYSTEM_PROMPT
    loss_system_prompt = tg.Variable(loss_system_prompt, requires_grad=False,
                                     role_description="system prompt to the loss function")

    instruction = INSTRUCTION

    format_string = "{instruction}\nProblem: {{problem}}\nCurrent Code: {{code}}"
    format_string = format_string.format(instruction=instruction)

    fields = {"problem": None, "code": None}
    formatted_llm_call = tg.autograd.FormattedLLMCall(engine=llm_engine,
                                                      format_string=format_string,
                                                      fields=fields,
                                                      system_prompt=loss_system_prompt)
    def loss_fn(problem: tg.Variable, code: tg.Variable) -> tg.Variable:
        inputs = {"problem": problem, "code": code}

        return formatted_llm_call(inputs=inputs,
                                  response_role_description=f"evaluation of the {code.get_role_description()}")

    loss = loss_fn(problem, code)
    loss.backward()
    optimizer.step()
    longest_increasing_subsequence = run_function_in_interpreter(code.value)
    eval_function_with_asserts(longest_increasing_subsequence)


def test_bugged():

    with pytest.raises(Exception):
        # bugged solution should throw an exception
        longest_increasing_subsequence = run_function_in_interpreter(BUGGED_SOLUTION)
        eval_function_with_asserts(longest_increasing_subsequence)

    llm_engine = tg.get_engine("experimental:gpt-4o-mini")
    tg.set_backward_engine(llm_engine, override=True)

    code = tg.Variable(value=BUGGED_SOLUTION,
                       requires_grad=True,
                       role_description="code instance to optimize")

    problem = tg.Variable(PROBLEM_TEXT,
                          requires_grad=False,
                          role_description="the coding problem")

    optimizer = tg.TGD(parameters=[code])

    loss_system_prompt = SYSTEM_PROMPT
    loss_system_prompt = tg.Variable(loss_system_prompt, requires_grad=False,
                                     role_description="system prompt to the loss function")

    instruction = INSTRUCTION

    format_string = "{instruction}\nProblem: {{problem}}\nCurrent Code: {{code}}"
    format_string = format_string.format(instruction=instruction)

    fields = {"problem": None, "code": None}
    formatted_llm_call = tg.autograd.FormattedLLMCall(engine=llm_engine,
                                                      format_string=format_string,
                                                      fields=fields,
                                                      system_prompt=loss_system_prompt)
    def loss_fn(problem: tg.Variable, code: tg.Variable) -> tg.Variable:
        inputs = {"problem": problem, "code": code}

        return formatted_llm_call(inputs=inputs,
                                  response_role_description=f"evaluation of the {code.get_role_description()}")

    loss = loss_fn(problem, code)
    loss.backward()
    optimizer.step()
    longest_increasing_subsequence = run_function_in_interpreter(code.value)
    eval_function_with_asserts(longest_increasing_subsequence)