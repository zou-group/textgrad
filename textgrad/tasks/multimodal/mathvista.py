import re
import io
import platformdirs
from PIL import Image

from textgrad.tasks.base import Dataset
from textgrad.loss import ImageQALoss
from textgrad.variable import Variable
try:
    from Levenshtein import distance
except ImportError:
    raise ImportError("Please install the Levenshtein package using 'pip install python-Levenshtein' to use mathvista.")

def compress_image(decoded_image, max_size_bytes=3.6*1024*1024):
    # Convert image to RGB if it's in a mode that JPEG does not support
    if decoded_image.mode not in ['RGB', 'L']:
        decoded_image = decoded_image.convert('RGB')

    buffer = io.BytesIO()
    decoded_image.save(buffer, format='JPEG')
    size = buffer.tell()
    
    if size <= max_size_bytes:
        buffer.seek(0)
        return buffer.getvalue()
    
    width, height = decoded_image.size
    while size > max_size_bytes:
        print(f"Compressing image to {width}x{height}...")
        width = int(width * 0.9)
        height = int(height * 0.9)
        resized_image = decoded_image.resize((width, height), Image.LANCZOS)
        
        buffer = io.BytesIO()
        resized_image.save(buffer, format='JPEG')
        size = buffer.tell()
        
        if width <= 1 or height <= 1:
            raise ValueError("Unable to compress image to the desired size without excessive loss of resolution")
    
    buffer.seek(0)
    return buffer.getvalue()


def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def extract_answer(response, problem, quick_extract=False):
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']
    pid = problem['pid']

    if response == "":
        return ""
    
    if question_type == 'multi_choice' and response in choices:
        return response
    
    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except:
            pass

    # quick extraction
    if quick_extract:
        print("Quickly extracting answer...")
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except:
            pass

    else:
        raise NotImplementedError("Extraction using LLMs are to-be-implemented.")


def get_most_similar(prediction, choices):
    """
    Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
    """
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]
    # return min(choices, key=lambda choice: distance(prediction, choice))


def normalize_extracted_answer(extraction, question_data):
    """
    Normalize the extracted answer to match the answer type
    """
    choices = question_data["choices"]
    question_type = question_data["question_type"]
    answer_type = question_data["answer_type"]
    precision = question_data["precision"]

    if question_type == 'multi_choice':
        # make sure the extraction is a string
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except:
                extraction = ""
    
        # extract "A" from "(A) text"
        letter = re.findall(r'\(([a-zA-Z])\)', extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()
        
        options = [chr(ord('A') + i) for i in range(len(choices))]
            
        if extraction in options:
            # convert option letter to text, e.g. "A" -> "text"
            ind = options.index(extraction)
            extraction = choices[ind]
        else:
            # select the most similar option
            extraction = get_most_similar(extraction, choices)
        assert extraction in choices

    elif answer_type == 'integer':
        try:
            extraction = str(int(float(extraction)))
        except:
            extraction = None

    elif answer_type == 'float':
        try:
            extraction = str(round(float(extraction), int(precision)))
        except:
            extraction = None

    elif answer_type == 'list':
        try:
            extraction = str(extraction)
        except:
            extraction = None

    return extraction
    

def safe_equal(prediction, answer):
    """
    Check if the prediction is equal to the answer, even if they are of different types
    """
    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        print(e)
        return False
    

class MathVistaDataset(Dataset):
    def __init__(self, evaluation_api:str, root: str=None, split: str="testmini", task_instruction: str=None, evaluation_instruction: str=None, *args, **kwargs):
        """MathVista dataset from HF."""
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        self.root = root
        assert split in ["testmini", "test"]
        self.data = load_dataset("AI4Math/MathVista", cache_dir=root, split=split)
        self.split = split
        self.evaluation_api = evaluation_api
        self.anwer_extraction_openai_engine = "gpt-3.5-turbo" # robust enough for answer extraction
        self.task_instruction = self.get_default_task_instruction(task_instruction) # NOTE: check the task instruction
        self.evaluation_instruction = self.get_default_evaluation_instruction(evaluation_instruction) # NOTE: check the evaluation instruction
    
    def __getitem__(self, index):
        row = self.data[index]
        pid = row["pid"]
        # question = row["question"]
        # image = row["image"]
        decoded_image = row["decoded_image"]
        choices = row["choices"]
        unit = row["unit"]
        precision = row["precision"]
        answer = row["answer"]
        question_type = row["question_type"]
        answer_type = row["answer_type"]
        metadata = row["metadata"] 
        query = row["query"]
        query = f"{self.task_instruction}\n{query}" # NOTE: Add the task description

        # NOTE: convert image to bytes
        if "claude" in self.evaluation_api.model_string:
            # TODO @lupantech This does not seem neat.
            image_bytes = compress_image(decoded_image)
        else:
            buffer = io.BytesIO()
            decoded_image.save(buffer, format='png')
            image_bytes = buffer.getvalue()
            buffer.close()

        # NOTE: ques_data stores other fields that might be useful later
        ques_data = {
            "pid": pid,
            "query": query,
            "choices": choices,
            "unit": unit,
            "precision": precision,
            "answer": answer,
            "question_type": question_type,
            "answer_type": answer_type,
            "metadata": metadata
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
            task_instruction = "You will answer a mathematical reasoning question based on an image. Please ensure you accurately interpret the image and think step by step."
        return task_instruction
    
    def get_default_evaluation_instruction(self, instruction):
        if instruction is not None:
            print("Using user-defined evaluation instruction:\n", instruction, "\n")
            evaluation_instruction = instruction
        else:
            evaluation_instruction = "Please evaluate the existing answer to the visual math problem without solving it yourself. Verify that the answer provides accurate reasoning logic to address the question."
        return evaluation_instruction

    def create_test_prompt(demo_prompt, query, response):
        demo_prompt = demo_prompt.strip()
        test_prompt = f"{query}\n\n{response}"
        full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
        return full_prompt

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
        extracted_answer  = extract_answer(response_text, question_data)

        # Normalize the extracted answer to match the answer type
        normalized_answer = normalize_extracted_answer(extracted_answer, question_data)

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
