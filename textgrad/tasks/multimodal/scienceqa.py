import platformdirs
import re
import os
import io
from collections import Counter
from PIL import Image

from textgrad.tasks.base import Dataset
from textgrad.loss import ImageQALoss
from textgrad.variable import Variable


def compress_image(decoded_image, max_size_bytes=3.6*1024*1024):
    # First, try saving as PNG without any compression
    buffer = io.BytesIO()
    decoded_image.save(buffer, format='PNG')
    size = buffer.tell()
    
    # If the original PNG is already small enough, return it
    if size <= max_size_bytes:
        buffer.seek(0)
        return buffer.getvalue()
    
    # If PNG is too large, resize the image
    width, height = decoded_image.size
    while size > max_size_bytes:
        print(f"Compressing image to {width}x{height}...")
        width = int(width * 0.9)
        height = int(height * 0.9)
        resized_image = decoded_image.resize((width, height), Image.LANCZOS)
        
        buffer = io.BytesIO()
        resized_image.save(buffer, format='PNG')
        size = buffer.tell()
        
        if width <= 1 or height <= 1:
            raise ValueError("Unable to compress image to the desired size without excessive loss of resolution")
    
    buffer.seek(0)
    return buffer.getvalue()

def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text).lower()
    return text

def find_most_similar_choice(text, choices):
    # Preprocess the given text
    text = preprocess_text(text)
    text_words = Counter(text.split())
    scores = []
    for choice in choices:
        choice_text = preprocess_text(choice)
        choice_words = Counter(choice_text.split())
        common_words = sum((text_words & choice_words).values()) # the number of common words
        scores.append(common_words)
    most_similar_index = scores.index(max(scores))
    return most_similar_index

def extract_answer(response_text):
    """
    Extract the answer from the response.
    For example, "xxxxx. Answer: (A) a gas." -> "A"
    If extraction fails, return the entire string after "Answer: ".
    """
    # Attempt to match the format "Answer: (A)" or "Answer: (a)" in case-sensitive manner
    match = re.search(r"Answer: \(([A-Z])\)", response_text)
    if match:
        return match.group(1).upper()  # Return as uppercase
    else:
        # Fallback: match the format "Answer: " followed by any characters until the next period or end of line
        fallback_match = re.search(r"Answer: ([^\.]+)", response_text)
        if fallback_match:
            return fallback_match.group(1).strip()
    return response_text

def normalize_extracted_answer(extracted_answer, question_data, options):
    # Normalize the extracted answer
    choices = question_data["choices"]
    options = options[:len(choices)]

    # 'A' -> one of the choices
    if extracted_answer in options:
        normalized_answer = options.index(extracted_answer)
        return normalized_answer

    # '(a) a gas.'
    for choice in choices:
        if choice.lower() in extracted_answer.lower():
            normalized_answer = choices.index(choice)
            return normalized_answer
        
    # find the most similar choice
    normalized_answer = find_most_similar_choice(extracted_answer, choices)
    return normalized_answer

def safe_equal(a, b):
    # Check if two intergers are equal
    return a == b

class ScienceQADataset(Dataset):
    def __init__(self, evaluation_api:str, root: str=None, split: str="test", task_instruction: str=None, evaluation_instruction: str=None, *args, **kwargs):
        """ScienceQA dataset from HF."""
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        self.root = root
        self.split = split
        assert split in ["test"]
        self.data = self.load_scienceqa_data()
        self.options = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        self.evaluation_api = evaluation_api
        self.task_instruction = self.get_default_task_instruction(task_instruction) # NOTE: check the task instruction
        self.evaluation_instruction = self.get_default_evaluation_instruction(evaluation_instruction) # NOTE: check the evaluation instruction
        
    def __getitem__(self, index):
        row = self.data[index]
        pid = row["pid"]
        image = row["image"]
        question = row["question"]
        choices = row["choices"]
        answer = row["answer"]
        hint = row["hint"]

        query = f"{self.task_instruction}" # NOTE: Add the task description
        if hint is not None and len(hint) > 0:
            query += f"\nContext: {hint}"
        query += f"\nQuestion: {question}"
        if choices:
            choice_list = []
            for i, c in enumerate(choices):
                choice_list.append("({}) {}".format(self.options[i], c))
            choice_txt = " ".join(choice_list)
            query += f"\nChoices: {choice_txt}"

        # NOTE: convert image to bytes
        if "claude" in self.evaluation_api.model_string:
            image_bytes = compress_image(image)
            # print("Image size:", len(image_bytes))
        else:
            buffer = io.BytesIO()
            image.save(buffer, format='png')
            image_bytes = buffer.getvalue()
            buffer.close()

        # NOTE: ques_data stores other fields that might be useful later
        ques_data = {
            "pid": pid,
            "question": question,
            "choices": choices,
            "hint": hint
        }
        test_time_objective = self._get_instance_test_time_objective(query, image_bytes)
        instance_eval_fn = self._get_instance_eval_fn(query, answer, ques_data)
        return image_bytes, query, answer, ques_data, test_time_objective, instance_eval_fn # NOTE: check the sample format

    def __len__(self):
        return len(self.data)

    def get_default_task_instruction(self, instruction):
        if instruction is not None:
            print("Using user-defined task instruction:\n", instruction, "\n")
            task_instruction = instruction
        else:
            task_instruction = "You will answer a scientific question based on an image. Please ensure you accurately interpret the image and think step by step. The last line of your answer should be formatted as follows: 'Answer: (X) Your Option.'"
        return task_instruction
    
    def load_scienceqa_data(self):
        scienceqa_dir = os.path.join(self.root, "scienceqa")
        try:
            from datasets import Dataset
            data = Dataset.load_from_disk(scienceqa_dir)
            print("Loaded ScienceQA dataset from cache.")
            return data
        except FileNotFoundError:
            from datasets import load_dataset
            data = load_dataset("derek-thomas/ScienceQA", split=self.split)
            data_img = data.filter(lambda x: x['image'] is not None) # select examples with a non-empty question image
            data_img = data_img.map(lambda x, i: {'pid': str(i), **x}, with_indices=True) # add index ID (string) for each example
            data_img.save_to_disk(scienceqa_dir)
            print("Loaded ScienceQA dataset from HF.")
            return data_img
        
    def get_default_evaluation_instruction(self, instruction):
        if instruction is not None:
            print("Using user-defined evaluation instruction:\n", instruction, "\n")
            evaluation_instruction = instruction
        else:
            evaluation_instruction = "Please evaluate the existing answer to the visual scientific problem without solving it yourself. Verify that the answer accurately understands the image, provides appropriate knowledge and reasoning logic to address the question."
        return evaluation_instruction

    def _get_instance_test_time_objective(self, question: str, image: bytes):
        """Define the loss function for the test time optimization."""
        eval_fn = ImageQALoss(evaluation_instruction=self.evaluation_instruction, engine=self.evaluation_api)
        def test_time_objective(instance: Variable):
            var_image = Variable(image, role_description="image input", requires_grad=False)
            var_question = Variable(question, role_description="question input", requires_grad=False)
            return eval_fn(question=var_question, image=var_image, response=instance)
        return test_time_objective
        
    def eval_extraction_and_matching(self, response_text, correct_answer, question_data):
        # Extract the precited answer text from the response
        extracted_answer  = extract_answer(response_text)

        # Normalize the extracted answer to match the answer type
        normalized_answer = normalize_extracted_answer(extracted_answer, question_data, self.options)

        # Verify the prediction is true or false
        true_false = safe_equal(normalized_answer, correct_answer)

        # Calculate the score and store the result data
        # NOTE: check the result data format
        score = 1 if true_false else 0
        result_data = {
            "extracted_answer": extracted_answer,
            "normalized_answer": normalized_answer,
            "true_false": true_false
        }
        return score, result_data

    def _get_instance_eval_fn(self, question_prompt: str, answer: str, ques_data: dict):
        """
        Define the evaluation function for scoring the response.
        Extraxct the short answer from the response and compare it with the ground truth.
        """
        # NOTE: check the evaluation function format
        eval_extraction_based_fn = lambda response: self.eval_extraction_and_matching(response.value, answer, ques_data)
        return eval_extraction_based_fn
