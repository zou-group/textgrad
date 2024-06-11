import re
import platformdirs
import random
from textgrad.variable import Variable
from textgrad.loss import MultiFieldTokenParsedEvaluation
from .base import Dataset
from textgrad.loss import MultiChoiceTestTime

# Below template is from https://github.com/openai/simple-evals/blob/main/common.py#L12
QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

def eval_string_based(response_text, correct_answer):
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"
    
    match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
    extracted_answer = match.group(1) if match else None
    score = 1.0 if extracted_answer == correct_answer else 0.0
    return score

class GPQA(Dataset):
    def __init__(self, subset:str, root: str=None, *args, **kwargs):
        """
        GPQA dataset from HF."""
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        self.root = root
        assert subset in ["gpqa_main", "gpqa_diamond", "gpqa_extended"]
        self.subset = subset
        self.data = load_dataset("Idavidrein/gpqa", subset, split="train", cache_dir=root)
        self._task_description = 'GPQA task' # Need to update
            
    def __getitem__(self, index):
        row = self.data[index]
        
        choices = [row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], row['Correct Answer']]
        choices = [choice.strip() for choice in choices]
        random.seed(42)
        random.shuffle(choices)
        choices_dict = dict(
                A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
            ) 
        correct_answer_idx = choices.index(row['Correct Answer'].strip())
        
        # Choices will be a. Choice 1 b. Choice 2 ... etc
        question_prompt = QUERY_TEMPLATE_MULTICHOICE.format(**choices_dict)
        answer = chr(65+correct_answer_idx)
        return question_prompt, answer

    def __len__(self):
        return len(self.data)

    def get_default_task_instruction(self):
        return "Given a multiple choice question, the goal is to select the correct answer from the choices."


class GPQAInstanceDataset(GPQA):
    def __init__(self, evaluation_api, subset:str, root: str=None, split: str="train", max_samples=-1):
        super().__init__(subset, root, split, max_samples)
        self.evaluation_api = evaluation_api

        
    def _get_instance_test_time_objective(self, question: str):
        evaluation_instruction = "Below is a multi-choice question and a prediction. You are an expert scientist. Your job is to investigate the prediction. Critically go through reasoning steps, and see if there is a reason why the prediction could be incorrect."
        evaluation_instruction += "\nUse the Janusian Process. Think about whether alternative answers could be true. Raise creative and critical objections to the solution, when needed."
        eval_fn = MultiChoiceTestTime(evaluation_instruction, engine=self.evaluation_api)
        def test_time_objective(instance: Variable):
            return eval_fn(question, instance)
        return test_time_objective
        

    def _legacy_get_instance_eval_fn(self, question_prompt: str, answer: str):
        role_descriptions = [
            "Question for the task",
            "Correct answer",
            "Solution and prediction from the language model"
        ]
        eval_system_prompt = Variable("You are a language model that evaluates the accuracy of a prediction for a mathematical question answering task. Only call a prediction accurate if it is the same as the ground truth answer.", requires_grad=False, role_description="system prompt for the evaluation")

        evaluation_instruction = "Below is a question from a question-answering task, the ground truth answer, and a prediction. Is the final prediction correct, i.e. the same as the ground truth answer? Say only 1 (yes) or 0 (no). Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        eval_instruction = Variable(evaluation_instruction, requires_grad=False, role_description="evaluation instruction for the task")
        eval_fn = MultiFieldTokenParsedEvaluation(
            eval_instruction,
            engine=self.evaluation_api,
            role_descriptions=role_descriptions,
            parse_tags=["<ACCURACY>", "</ACCURACY>"],
            system_prompt=eval_system_prompt
        )
        
        answer_var = Variable(answer, requires_grad=False, role_description="Correct answer")
        question_var = Variable(question_prompt, requires_grad=False, role_description="Question for the task")
        def instance_eval_fn(instance):
            eval_output = eval_fn([question_var, answer_var, instance])
            return eval_fn.parse_output(eval_output)
        return instance_eval_fn
    
    
    def _get_instance_eval_fn(self, question_prompt: str, answer: str):
        eval_string_based_fn = lambda response: eval_string_based(response.value, answer)
        return eval_string_based_fn
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        
        choices = [row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], row['Correct Answer']]
        choices = [choice.strip() for choice in choices]
        random.seed(42)
        random.shuffle(choices)
        choices_dict = dict(
                A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
            ) 
        correct_answer_idx = choices.index(row['Correct Answer'].strip())
        
        # Choices will be a. Choice 1 b. Choice 2 ... etc
        question_prompt = QUERY_TEMPLATE_MULTICHOICE.format(**choices_dict)
        answer = chr(65+correct_answer_idx)

        # TODO: Make the two-way comparison class abstract enough.
        # TODO: How do we determine the role of the instances? We should be more consistent
        return question_prompt, answer, self._get_instance_test_time_objective(question_prompt), self._get_instance_eval_fn(question_prompt, answer)

    def get_task_description(self):
        return "Given a multiple choice question, the goal is to select the correct final answer from the choices."



class GPQAInstanceDatasetOpenAI(Dataset):
    def __init__(self, evaluation_api, subset:str, root: str=None, *args, **kwargs):
        """
        GPQA dataset from OpenAI (from https://github.com/openai/simple-evals/)"""
        import pandas as pd
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        self.root = root
        assert subset in ["gpqa_main", "gpqa_diamond", "gpqa_extended"]
        self.subset = subset
        df = pd.read_csv(f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{subset[5:]}.csv")
        examples = [row.to_dict() for _, row in df.iterrows()]
        rng = random.Random(0)
        self.data = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]
        self._task_description = 'GPQA task' # Need to update
        self.evaluation_api = evaluation_api

    
    def _get_instance_test_time_objective(self, question: str):
        evaluation_instruction = "Below is a multi-choice question and a prediction. You are a critical and creative scientist. Your job is to investigate the prediction. Critically go through reasoning steps, and see if there is a reason why the prediction could be incorrect."
        evaluation_instruction = "\nUse the Janusian Process, think about whether alternative answers could be true."
        eval_fn = MultiChoiceTestTime(evaluation_instruction, engine=self.evaluation_api)
        def test_time_objective(instance: Variable):
            return eval_fn(question, instance)
        return test_time_objective
        

    def _legacy_get_instance_eval_fn(self, question_prompt: str, answer: str):
        role_descriptions = [
            "Question for the task",
            "Correct answer",
            "Solution and prediction from the language model"
        ]
        eval_system_prompt = Variable("You are a language model that evaluates the accuracy of a prediction for a mathematical question answering task. Only call a prediction accurate if it is the same as the ground truth answer.", requires_grad=False, role_description="system prompt for the evaluation")

        evaluation_instruction = "Below is a question from a question-answering task, the ground truth answer, and a prediction. Is the final prediction correct, i.e. the same as the ground truth answer? Say only 1 (yes) or 0 (no). Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        eval_instruction = Variable(evaluation_instruction, requires_grad=False, role_description="evaluation instruction for the task")
        eval_fn = MultiFieldTokenParsedEvaluation(
            eval_instruction,
            engine=self.evaluation_api,
            role_descriptions=role_descriptions,
            parse_tags=["<ACCURACY>", "</ACCURACY>"],
            system_prompt=eval_system_prompt
        )
        
        answer_var = Variable(answer, requires_grad=False, role_description="Correct answer")
        question_var = Variable(question_prompt, requires_grad=False, role_description="Question for the task")
        def instance_eval_fn(instance):
            eval_output = eval_fn([question_var, answer_var, instance])
            return eval_fn.parse_output(eval_output)
        return instance_eval_fn
    
    
    def _get_instance_eval_fn(self, question_prompt: str, answer: str):
        eval_string_based_fn = lambda response: eval_string_based(response.value, answer)
        return eval_string_based_fn
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        
        choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
        choices = [choices[i] for i in row["permutation"]]
        correct_answer_idx = choices.index(row["Correct Answer"])
        answer = "ABCD"[correct_answer_idx]
        choices_dict = dict(
            A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
        )
        
        # Choices will be a. Choice 1 b. Choice 2 ... etc
        question_prompt = QUERY_TEMPLATE_MULTICHOICE.format(**choices_dict)

        # TODO: Make the two-way comparison class abstract enough.
        # TODO: How do we determine the role of the instances? We should be more consistent
        return question_prompt, answer, self._get_instance_test_time_objective(question_prompt), self._get_instance_eval_fn(question_prompt, answer)

    def get_default_task_instruction(self):
        return "Given a multiple choice question, the goal is to select the correct final answer from the choices."
