"""
This codes runs a code generation pipeline for instance optimization.

The dataset used is the LeetCodeHardEval dataset. Part of the LeetCodeHard-Gym dataset.

Since we are evaluating using leetcode, the code has to run in two phases:

In the first phase, we parallel process the generation of the code using OpenAI's APIs.
In the second phase, we evaluate the code using the leetcode environment.

We need to do this because we cannot parallelize the leeetcode evaluation as it makes API calls for which
we have a rate limit.


"""
import copy
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
import multiprocessing
from tqdm import tqdm
import textgrad
from textgrad.engine import get_engine
from textgrad import Variable
from textgrad.optimizer import TextualGradientDescent
from textgrad.tasks import load_instance_task
from prompts import CodeTestTimewithTests, SYSTEM_PROMPT_FOR_FIRST_CODE, CODE_INSTANCE_ROLE_DESCRIPTION
from evaluators.lt_eval import LeetCodeEvaluator
from evaluators.py_eval import PythonEvaluator

internal_evaluator = LeetCodeEvaluator()

def optimization_one_iteration(optimizer, instance_var, prompt, tests):
    """
    This is a single iteration of optimization
    :param optimizer:
    :param instance_var:
    :param prompt:
    :return:
    """
    pt = PythonEvaluator()
    tests = tests.split("\n")

    # Evaluate the code
    # state is True/False for each test
    state, test_string = pt.evaluate(instance_var.value, tests)

    optimizer.zero_grad()
    loss_fn = CodeTestTimewithTests(engine=ENGINE_API)

    test_time_loss = loss_fn(prompt, instance_var, test_string)

    test_time_loss.backward()
    optimizer.step()

    return state, test_string


def generate_starting_solution(prompt):
    """
    This is the first attempt at solving the problem.
    :param prompt:
    :return:
    """
    llm_first_code = ENGINE_API.generate(prompt, system_prompt=SYSTEM_PROMPT_FOR_FIRST_CODE)

    llm_first_code = llm_first_code.split("```python")[1].split("```")[0]
    return llm_first_code


def evaluation_and_optimization_pipeline(task_id, prompt, index, tests, MAX_ITERS):
    """
    :param prompt:
    :param index:
    :return:
    """
    print("Start of optimization pipeline", index)

    generated_programs = []
    gpt_4_first_code = generate_starting_solution(prompt)
    n_iter = 0
    generated_programs.append({"code": gpt_4_first_code,
                               "gradients": None,
                               "task_id": task_id,
                               "state": None,
                               "test_string": None,
                               "iteration": n_iter,
                               })

    instance_var = Variable(gpt_4_first_code, requires_grad=True,
                            role_description=CODE_INSTANCE_ROLE_DESCRIPTION)

    optimizer = TextualGradientDescent(engine=ENGINE_API,
                                       parameters=[instance_var],
                                       constraints=["Do not add asserts to the code",
                                                    "Code must contain imports"])

    pt = PythonEvaluator()
    # Evaluate the code
    # state is True/False for each test
    state, test_string = pt.evaluate(gpt_4_first_code, tests.split("\n"))

    generated_programs[-1]["state"] = state
    generated_programs[-1]["test_string"] = test_string
    # if all test passed we early stop

    for iter in range(1 + MAX_ITERS):
        state, test_string = optimization_one_iteration(optimizer, instance_var, prompt, tests)
        generated_programs[-1]["state"] = state
        generated_programs[-1]["test_string"] = test_string
        # if all test passed we early stop
        if ((iter != 0) and all(state)):
            break

        n_iter += 1
        generated_programs.append({"code": instance_var.value,
                                   "gradients": str(instance_var.gradients),
                                   "task_id": task_id,
                                   "state": None,
                                   "test_string": None,
                                   "iteration": n_iter,
                                   })


    print("End of optimization pipeline", index)

    # we get the first program (zero-shot) and the last program (after optimization, if it exists)
    return generated_programs


def evaluate_or_except(program):
    try:
        res = (internal_evaluator.check_if_in_cache_or_submit(program["task_id"], program["code"]))
        return res[0], res[1], res[2], res[3]
    except Exception as e:
        print(e)
        return False, -1, -1, -1


if __name__ == "__main__":

    SEEDS = [55]#, 91, 17, 21]#, 34]

    collection = {
        "task_id": [],
        "problem": [],
        "seed": [],
        "success": [],
        "test_cases_passed": [],
        "test_cases_total": [],
        "runtime": [],
        "local_tests": [],
        "local_test_state": [],
        "iteration_idx": []
    }


    for seed in SEEDS:
        instance_task = "LeetCodeHardEval"

        TEST_ENGINE = "gpt-4o"
        ENGINE_API = get_engine(engine_name=TEST_ENGINE, seed=seed)
        MAX_PROGRAMS_TO_OPTIMIZE = 39
        MULTIPROC_POOLS = min([20, MAX_PROGRAMS_TO_OPTIMIZE, multiprocessing.cpu_count()])
        MAX_ITERS = 4

        textgrad.set_backward_engine(ENGINE_API, override=True)

        dataset = load_instance_task(instance_task, ENGINE_API)

        code_dataset_examples = [(task_id, prompt, index, tests, MAX_ITERS)
                                 for index, (task_id, prompt, tests)
                                 in enumerate(dataset)][:MAX_PROGRAMS_TO_OPTIMIZE]

        with multiprocessing.Pool(MULTIPROC_POOLS) as pool:
            programs_and_gradients = pool.starmap(evaluation_and_optimization_pipeline, code_dataset_examples)

        all_programs = []
        zero_programs = []

        ctr = 0
        for list_of_programs in tqdm(programs_and_gradients):
            list_of_programs = [list_of_programs[0], list_of_programs[-1]]
            for iter, program in enumerate(list_of_programs):
                res = evaluate_or_except(program)
                program["evaluation"] = res[0]
                program["total_correct"] = res[1]
                program["total_tests"] = res[2]
                program["runtime"] = res[3]
                program["iteration_idx"] = iter
                all_programs.append(program)
                if ((program['iteration_idx'] == 0)):
                    ctr += 1
                    zero_programs.append(program)

        print("CounterZero", len(zero_programs))
        ctr = 0
        for program in all_programs:
            if program["iteration_idx"] == 0:
                ctr += 1
        print("CounterAll", ctr)
        
        for program in all_programs:
            collection["task_id"].append(program["task_id"])
            collection["problem"].append(program["code"])
            collection["seed"].append(seed)
            collection["iteration_idx"].append(program['iteration_idx'])
            collection["success"].append(program["evaluation"])
            collection["test_cases_passed"].append(program["total_correct"])
            collection["test_cases_total"].append(program["total_tests"])
            collection["runtime"].append(program["runtime"])
            collection["local_tests"].append(program["test_string"])
            collection["local_test_state"].append(program["state"])
            

        program_df = pd.DataFrame(collection)

        program_df.to_csv(f"../results/code_optimization/textgrad/leetcode_supervised_{TEST_ENGINE}.csv", index=False)

