import re
import os
import json
import argparse
import concurrent
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
load_dotenv(override=True)
from collections import Counter

import textgrad as tg
from textgrad.tasks.multimodal import load_multimodal_instance_task

def load_checkpoint_results(checkpoint_file):
    all_solutions = {}
    if checkpoint_file.endswith(".jsonl"):
        with open(checkpoint_file, "r") as f:
            for line in f:
                data = json.loads(line)
                all_solutions[data["ques_data"]["pid"]] = data
    elif checkpoint_file.endswith(".json"):
        with open(checkpoint_file, "r") as f:
            all_solutions = json.load(f)
    print("Loaded existing results from", checkpoint_file)
    print(f"There are {len(all_solutions)} samples in the file!")
    return all_solutions

class MajorityVoting:
    def __init__(self, solutions, max_iterations):
        self.solutions = solutions
        self.max_iterations = max_iterations

    def majority_element(self, predictions):
        count = Counter(predictions)
        return max(count.keys(), key=count.get)

    def compute_accuracy(self):
        final_score = 0
        for n in range(1, self.max_iterations + 1):
            correct = 0
            for _, solution in self.solutions.items():
                answer = solution["answer"]
                predictions = [res["normalized_answer"] for res in solution["result_data"]]
                predictions = predictions[1:n+1]
                majority = self.majority_element(predictions)
                if majority == answer:
                    correct += 1
            acc = round(correct / len(self.solutions), 3)
            if acc > final_score:
                final_score = acc
        return final_score

def config():
    parser = argparse.ArgumentParser(description="Optimize a prompt for a task.")
    parser.add_argument("--task", type=str, default="mathvista", choices = ["mathvista", "scienceqa"], help="The task to evaluate the model on.")
    parser.add_argument("--multimodal", action='store_true', help="Determine if we want to load multimodal dataset.")
    parser.add_argument("--task_instruction", type=str, default=None, help="The instruction for the task.")
    parser.add_argument("--evaluation_instruction", type=str, default=None, help="The instruction for the evaluation.")
    parser.add_argument("--instance_role", type=str, default=None, help="The role description of the instance.")
    parser.add_argument("--image_role", type=str, default=None, help="The role description of the image.")
    parser.add_argument("--question_role", type=str, default=None, help="The role description of the question.")
    parser.add_argument("--cache_root", type=str, default=None, help="The cache directory to save data.")
    parser.add_argument("--results_dir", type=str, default="./results/solution_optimization_mm", help="The directory to save the results.")
    parser.add_argument("--output_file", type=str, default="", help="The output file to save or load the results.")
    parser.add_argument("--save_every", type=int, default=50, help="The frequency to save the results.")
    parser.add_argument("--engine", type=str, default="gpt-4o", help="The API to use for inference.")
    parser.add_argument("--eval_engine", type=str, default="gpt-4o", help="The API to use for evaluation.")
    parser.add_argument("--max_iterations", type=int, default=3, help="The maximum number of iterations of test-time updates.")
    parser.add_argument("--num_threads", type=int, default=16, help="The number of threads to use for evaluation.")
    parser.add_argument("--test_num", type=int, default=-1, help="The number of test samples to evaluate., -1 means all.")
    parser.add_argument("--majority_voting", action='store_true', default=False, help="Determine if we want to use majority voting.")
    return parser.parse_args()

def get_zeroshot_answer(question, mm_data=None):
    """Getting the zero-shot answer from an LLM without optimizing the response at test time."""
    # The system prompt is from: https://github.com/openai/simple-evals/blob/main/sampler/chat_completion_sampler.py
    STARTING_SYSTEM_PROMPT = (
        "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
        + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-07-01"
    )
    if args.question_role is not None:
        question_role = args.question_role
    else:
        question_role = "question to the language model"
    if args.image_role is not None:
        image_role = args.image_role
    else:
        image_role = "image associate with question to the language model"
    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, requires_grad=False, role_description="system prompt to the language model")
    if mm_data is None: # vanilla text.
        model = tg.BlackboxLLM(llm_engine, system_prompt)
        response = model(tg.Variable(question, requires_grad=False, role_description=question_role))
    else: # multi-modal (such as image).
        from textgrad.autograd import MultimodalLLMCall
        variable_mm_data = tg.Variable(mm_data, requires_grad=False, role_description=image_role)
        variable_question = tg.Variable(question, requires_grad=False, role_description=question_role)
        response = MultimodalLLMCall(args.engine)([variable_mm_data, variable_question])
    return response

def run_test_time_training(sample):
    performance_history = []
    results_data = []
    mm_data, question, answer, ques_data, test_time_objective, instance_eval_fn = sample # NOTE: check the sample format
    zero_shot_response = get_zeroshot_answer(question, mm_data)

    # NOTE: Set the instance role description
    if args.instance_role is not None:
        instance_role_description = args.instance_role
    else:
        if args.task == "mathvista":
            instance_role_description = "detailed reasoning and prediction for the image-related math question."
        elif args.task == "scienceqa":
            instance_role_description = "detailed reasoning and prediction for the image-related science question."
        else:
            instance_role_description = "precise and short answer to the image-related question."

    instance_var = tg.Variable(zero_shot_response.value, requires_grad=True, role_description=instance_role_description)
    
    # Evaluate the zero-shot response
    # NOTE: The instance_eval_fn should return a tuple of (score, result_data) if result_data is needed.
    eval_score = instance_eval_fn(instance_var)
    if isinstance(eval_score, tuple):
        score, result_data = eval_score
    else:
        score = eval_score
        result_data = None
    performance_history.append(int(score))
    results_data.append(result_data)
    
    optimizer = tg.TextualGradientDescent(engine=llm_engine, 
                                        parameters=[instance_var], 
                                        constraints=["Your response should answer the question given an image."])

    predictions = []
    predictions.append(tg.Variable(
        instance_var.value,
        role_description=instance_var.role_description
        ))
    
    loss_history = []
    
    # Start test time optimization
    for _ in range(args.max_iterations):
        optimizer.zero_grad()
        # Compute the test time loss
        test_time_loss = test_time_objective(instance_var)
        loss_history.append(str(test_time_loss))
        test_time_loss.backward()
        optimizer.step()

        eval_score = instance_eval_fn(instance_var)
        # NOTE: The instance_eval_fn should return a tuple of (score, result_data) if result_data is needed.
        if isinstance(eval_score, tuple):
            score, result_data = eval_score
        else:
            score = eval_score
            result_data = None

        predictions.append(tg.Variable(
            instance_var.value,
            role_description=instance_var.role_description
        ))
        performance_history.append(int(score))
        results_data.append(result_data)

    return question, answer, predictions, performance_history, loss_history, results_data, ques_data # NOTE: check the return format

args = config()
llm_engine = tg.get_engine(engine_name=args.engine)
tg.set_backward_engine(llm_engine, override=True)

eval_engine = tg.get_engine(engine_name=args.eval_engine)
test_set = load_multimodal_instance_task(args.task,
                                         evaluation_api=eval_engine, 
                                         task_instruction=args.task_instruction, # NOTE: check the instruction
                                         evaluation_instruction=args.evaluation_instruction, # NOTE: check the instruction
                                         root=args.cache_root)

# Create the results directory if it does not exist
os.makedirs(args.results_dir, exist_ok=True)
output_file = f"{args.task}_predictions.json" if args.output_file == "" else args.output_file
output_file = os.path.join(args.results_dir, output_file)
output_jsonl_file = output_file.replace(".json", ".jsonl")

# Read the checkpoint result file if it exists
if os.path.exists(output_jsonl_file):
    all_solutions = load_checkpoint_results(output_jsonl_file)
elif os.path.exists(output_file):
    all_solutions = load_checkpoint_results(output_file)
else:
    all_solutions = {}
all_history = [solution["performance_history"] for solution in all_solutions.values()]

if args.num_threads > 0:
    # Use ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        print("\nPreparing to run test-time training...")
        for i, sample in enumerate(test_set):
            image_bytes, query, answer, ques_data, test_time_objective, instance_eval_fn = sample # NOTE: check the sample format
            pid = ques_data["pid"] # NOTE: check the question id
            if pid in all_solutions:
                # print(f"Skipping sample {pid}")
                continue
            if args.test_num > 0 and i >= args.test_num:
                break
            future = executor.submit(run_test_time_training, sample)
            futures.append(future)

        print("\nRunning test-time training...")
        for _, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)):
            question, answer, predictions, performance_history, loss_history, result_data, ques_data = future.result() # NOTE: check the return format
            pid = ques_data["pid"] # NOTE: check the question id
            all_solutions[pid] = {
                "question": question,
                "answer": answer, 
                "predictions": [p.value for p in predictions], 
                "loss_history": loss_history,
                "performance_history": performance_history,
                "result_data": result_data,
                "ques_data": ques_data
                }
            all_history.append(performance_history)
            # save result to output file in append mode
            with open(output_jsonl_file, "a") as f:
                json_line = json.dumps(all_solutions[pid])
                f.write(json_line + "\n")
else:
    # Use regular for loop
    if args.test_num > 0:
        total = min(args.test_num, len(test_set))
    else:
        total = len(test_set)
    for i, sample in enumerate(tqdm(test_set, total=total, position=0)):
        image_bytes, query, answer, ques_data, test_time_objective, instance_eval_fn = sample # NOTE: check the sample format
        pid = ques_data["pid"] # NOTE: check the question id
        if pid in all_solutions:
            continue
        if args.test_num > 0 and i >= args.test_num:
            break
        future = run_test_time_training(sample)

        question, answer, predictions, performance_history, loss_history, result_data,  ques_data = future # NOTE: check the return format
        result = {
            "question": question,
            "answer": answer,
            "predictions": [p.value for p in predictions],
            "performance_history": performance_history,
            "loss_history": loss_history,
            "result_data": result_data,
            "ques_data": ques_data
            }
        all_solutions[pid] = result
        all_history.append(performance_history)

        # save result to output file in append mode
        with open(output_jsonl_file, "a") as f:
            json_line = json.dumps(result)
            f.write(json_line + "\n")
        if i % args.save_every == 0:
            with open(output_file, "w") as f:
                print("Saving results to", output_file)
                json.dump(all_solutions, f, indent=4)

# Quick summary of the results
all_solutions = {k: all_solutions[k] for k in sorted(all_solutions, key=lambda x: int(x))} # sorted by int(pid)
with open(output_file, "w") as f:
    json.dump(all_solutions, f, indent=4)

print("\nSummary of the results:")
score_history = np.array(all_history).mean(axis=0)
score_history = [round(float(s), 3) for s in score_history]
print(score_history)

if args.majority_voting:
    print("\nMajority Voting:")
    majority_voting = MajorityVoting(all_solutions, args.max_iterations)
    final_score = majority_voting.compute_accuracy()
else:
    final_score = round(max(np.array(all_history).mean(axis=0)), 3)

print(f"Final Accuracy: {final_score}")
print(f"Total samples: {len(all_solutions)}")
print("Done!")
