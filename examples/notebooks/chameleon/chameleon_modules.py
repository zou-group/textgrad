import os
import re
import copy
import json
import numpy as np

from textgrad.engine.openai import ChatOpenAI

from typing import Dict, Callable, Any, List

import textgrad as tg
from textgrad.autograd import MultimodalLLMCall, LLMCall
from textgrad.autograd.string_based_ops import BackwardContext, StringBasedFunction
from textgrad.loss import ImageQALoss


# PlannerModule
class PlannerModule(StringBasedFunction):
    def __init__(self, llm_engine):
        self.llm_engine = llm_engine
        super().__init__(
            fn=self.forward, function_purpose="planning the sequence of modules"
        )

    def forward(self, question_text, metadata, *args, **kwargs):
        test_prompt, full_prompt = self._build_prompt(question_text, metadata)

        modules_variable = self.llm_engine(full_prompt)

        modules_variable = tg.Variable(
            modules_variable.get_value(),
            role_description="modules predicted by the policy module (planner).",
            requires_grad=True,
        )

        modules_variable.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward_static,
                response=modules_variable,
                function_purpose="plan the sequence of modules to execute.",
                inputs={"planner_prompt": full_prompt},
            )
        )

        modules = self._update_modules(modules_variable.get_value())
        return modules, modules_variable

    def _build_prompt(self, question_text, metadata):
        from prompts import prompt_policy

        demo_prompt = prompt_policy.prompt.strip()
        test_prompt = f"Question: {question_text}\n\nMetadata: {metadata}\n\nModules: "
        full_prompt = demo_prompt + "\n\n" + test_prompt

        full_prompt_variable = tg.Variable(
            full_prompt,
            role_description="instructions and information for the planner module.",
            requires_grad=False,
        )
        return test_prompt, full_prompt_variable

    def _update_modules(self, modules_string):
        valid_modules = [
            "image_captioner",
            "text_detector",
            "knowledge_retrieval",
            "solution_generator",
            "answer_generator",
        ]

        def find_modules(string):
            match = re.search(r"\[.*?\]", string)
            if match:
                modules_text = match.group(0)
                try:
                    modules_list = eval(modules_text)
                    if isinstance(modules_list, list):
                        return modules_list
                except (SyntaxError, NameError):
                    pass
            return []

        try:
            modules = find_modules(modules_string)
            modules = [module.lower().strip() for module in modules]
            modules = [module for module in modules if module in valid_modules]
            if modules[-2:] != ["solution_generator", "answer_generator"]:
                raise ValueError("Last two modules do not match default sequence")
        except:
            modules = ["solution_generator", "answer_generator"]

        return modules

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# TextDetectorModule
class TextDetectorModule(StringBasedFunction):
    def __init__(self):
        # load cache and model for text detection
        self.ocr_file = "ocr_cache.json"
        self.ocr_cache = self._load_cache(self.ocr_file)
        self.text_detector_model = self._build_text_detector_model()
        super().__init__(fn=self.forward, function_purpose="detecting text in an image")

    def _load_cache(self, file_path):
        """Helper function to load cache from a JSON file."""
        if os.path.exists(file_path):
            print(f"Loading cache from {file_path}")
            with open(file_path, "r") as file:
                return json.load(file)
        return {}

    def _build_text_detector_model(self):
        """Builds and returns the EasyOCR reader model."""
        try:
            import easyocr
            import platformdirs
        except ImportError:
            raise ImportError(
                "Please install the EasyOCR package using 'pip install easyocr' to use chameleon."
            )
        # load the model
        root = platformdirs.user_cache_dir("textgrad")
        return easyocr.Reader(["en"], model_storage_directory=root)

    def _get_or_check_for_ocr_cache(self, pid, image_bytes, example):
        # Execute the text detector model
        if "ocr" not in example:
            if pid in self.ocr_cache:
                texts = self.ocr_cache[pid]
            else:
                texts = self._detect_text(image_bytes)
                texts = [
                    ([[int(item) for item in sublist] for sublist in polygon], label)
                    for polygon, label, score in texts
                ]
                self.ocr_cache[pid] = texts
                json.dump(self.ocr_cache, open(self.ocr_file, "w"))
            example["ocr"] = str(texts)

        try:
            texts = example["ocr"]
        except Exception as e:
            print(f"An error occurred during OCR evaluation: {e}")
            texts = []

        return texts

    def _detect_text(self, image_bytes):
        """Detects text in the given image bytes using EasyOCR."""
        try:
            return self.text_detector_model.readtext(image_bytes)
        except Exception as e:
            print(f"Error detecting text: {e}")
            return []

    def forward(
        self, pid, image_bytes, tool_tape, example, last_output, *args, **kwargs
    ):
        texts = self._get_or_check_for_ocr_cache(pid, image_bytes, example)

        if not isinstance(tool_tape, tg.Variable):
            raise ValueError("tool_tape should be a Variable object.")

        texts = f"Detected text in the image: {texts}"
        text_response = tg.Variable(
            texts,
            role_description="text detected from the image using a tool.",
            predecessors=[last_output],
            requires_grad=True,
        )

        text_response.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward_static,
                response=text_response,
                function_purpose="detect text in the images",
                inputs={last_output.role_description: last_output},
            )
        )

        return text_response

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# KnowledgeRetrievalModule
class KnowledgeRetrievalModule(StringBasedFunction):
    def __init__(self, llm_engine):
        self.llm_engine = llm_engine
        super().__init__(
            fn=self.forward, function_purpose="retrieving relevant knowledge"
        )

    def forward(self, question_text, metadata, tool_tape, last_output, *args, **kwargs):
        test_prompt, full_prompt = self._build_prompt(
            question_text, metadata, tool_tape
        )

        knowledge = self.llm_engine(full_prompt)

        knowledge_text = f"\n\nKnowledge:\n{knowledge}"
        knowledge_variable = tg.Variable(
            knowledge_text,
            role_description="knowledge retrieved by the knowledge retrieval module.",
            predecessors=[last_output],
            requires_grad=True,
        )

        knowledge_variable.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward_static,
                response=knowledge_variable,
                function_purpose="retrieve knowledge for the query",
                inputs={last_output.role_description: last_output},
            )
        )

        return knowledge_variable

    def _build_prompt(self, question_text, metadata, tool_tape):
        from prompts import prompt_kr

        demo_prompt = prompt_kr.prompt.strip()
        if tool_tape.get_value() != "":
            test_prompt = f"Question: {question_text}\n\nMetadata: {metadata}\n\n{tool_tape.get_value()}\n\nKnowledge:\n"
        else:
            test_prompt = (
                f"Question: {question_text}\n\nMetadata: {metadata}\n\nKnowledge:\n"
            )
        full_prompt = demo_prompt + "\n\n" + test_prompt
        return test_prompt, full_prompt

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ImageCaptionerModule
class ImageCaptionerModule(StringBasedFunction):
    def __init__(self, local_mm_engine_name):
        self.local_mm_engine_name = local_mm_engine_name
        self.captioner_model = self._build_image_captioner_model()
        super().__init__(
            fn=self.forward, function_purpose="generating caption for an image"
        )

    def _build_image_captioner_model(self):
        """Builds and returns the image captioner model."""
        try:
            local_mm_engine = ChatOpenAI(
                model_string=self.local_mm_engine_name, is_multimodal=True
            )
            # print("Local OpenAI multimodal engine initialized.")
            return local_mm_engine
        except Exception as e:
            print(f"Error initializing OpenAI engine: {e}")
            return None

    def _generate_caption(self, image_bytes):
        prompt = "Describe the image."
        try:
            content = [prompt, image_bytes]
            caption = self.captioner_model(content)
            return caption
        except Exception as e:
            print(f"Error generating caption: {e}")
            return ""
        
    def forward(self, image_bytes, last_output, tool_tape, *args, **kwargs):
        caption = self._generate_caption(image_bytes)

        caption_text = f"\n\nImage caption: {caption}"
        caption_variable = tg.Variable(
            caption_text,
            role_description="caption generated by the image captioner module.",
            predecessors=[last_output],
            requires_grad=True,
        )

        caption_variable.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward_static,
                response=caption_variable,
                function_purpose="captioning an image",
                inputs={last_output.role_description: last_output},
            )
        )

        return caption_variable

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# SolutionGeneratorModule
class SolutionGeneratorModule(StringBasedFunction):
    def __init__(self, llm_engine):
        self.llm_engine = llm_engine
        super().__init__(fn=self.forward, function_purpose="generating solution")

    def forward(
        self, question_text, metadata, tool_tape, tool_tape_variables, *args, **kwargs
    ):
        test_prompt, full_prompt = self._build_prompt(
            question_text, metadata, tool_tape
        )

        solution = self.llm_engine(full_prompt)

        solution_text = f"\n\nSolution: {solution}"
        solution_variable = tg.Variable(
            solution_text,
            role_description="solution generated by the solution generator module.",
            predecessors=tool_tape_variables,
            requires_grad=True,
        )

        solution_variable.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward_static,
                response=solution_variable,
                function_purpose="generate solution using a set of tool calls",
                inputs={v.role_description: v for v in tool_tape_variables},
            )
        )

        return solution_variable

    def _build_prompt(self, question_text, metadata, tool_tape):
        from prompts import prompt_sg

        demo_prompt = prompt_sg.prompt_chameleon.strip()
        if tool_tape.get_value() != "":
            test_prompt = f"Question: {question_text}\n\nMetadata: {metadata}\n\n{tool_tape.get_value()}\n\nSolution: "
        else:
            test_prompt = (
                f"Question: {question_text}\n\nMetadata: {metadata}\n\nSolution: "
            )
        full_prompt = demo_prompt + "\n\n" + test_prompt
        return test_prompt, full_prompt

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# AnswerGeneratorModule
class AnswerGeneratorModule(StringBasedFunction):
    def __init__(self):
        super().__init__(fn=self.forward, function_purpose="generating final answer")

    def forward(self, solution, example, *args, **kwargs):
        prediction = self._generate_answer(solution.get_value(), example["choices"])

        prediction_variable = tg.Variable(
            prediction,
            role_description="prediction generated by the answer generator module.",
            predecessors=[solution],
            requires_grad=True,
        )

        prediction_variable.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward_static,
                response=prediction_variable,
                function_purpose="generate final answer",
                inputs={solution.role_description: solution},
            )
        )

        return prediction_variable

    def _generate_answer(self, output, options):
        inds = ["A", "B", "C", "D", "E"]
        success = False
        if output:
            pattern = re.compile(r"[Tt]he answer is ([A-Z])")
            res = pattern.findall(output)
            if len(res) > 0:
                ans = res[0]
                if ans in inds[: len(options)]:
                    success = True
                    prediction = options[inds.index(ans)]

        if not success:
            prediction = self._normalize_prediction(output, options)

        return prediction

    def _normalize_prediction(self, prediction, options):
        options = [x.lower() for x in options]
        if prediction is None:
            prediction = options[0]
        elif isinstance(prediction, str):
            if prediction not in options:
                scores = [self._score_string_similarity(x, prediction) for x in options]
                max_idx = int(np.argmax(scores))
                prediction = options[max_idx]
        return prediction

    def _score_string_similarity(self, str1, str2):
        if str1 == str2:
            return 2.0
        elif " " in str1 or " " in str2:
            str1_split = str1.split(" ")
            str2_split = str2.split(" ")
            overlap = list(set(str1_split) & set(str2_split))
            return len(overlap) / max(len(str1_split), len(str2_split))
        else:
            return 0.0

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ChameleonAgent:
    def __init__(self, args):
        self.args = args

        # Initialize LLM engine
        self.local_mm_engine_name = args.engine
        self.global_llm_engine = ChatOpenAI(
            model_string=args.engine, is_multimodal=False
        )
        self.global_llm_call = LLMCall(engine=self.global_llm_engine)

        # Initialize modules
        self.text_detector = TextDetectorModule()
        self.image_captioner = ImageCaptionerModule(self.local_mm_engine_name)
        self.knowledge_retrieval = KnowledgeRetrievalModule(self.global_llm_engine)
        self.solution_generator = SolutionGeneratorModule(self.global_llm_engine)
        self.answer_generator = AnswerGeneratorModule()
        self.planner = PlannerModule(self.global_llm_call)

        # Module registry
        self.module_registry: Dict[str, Callable] = {
            "text_detector": self.text_detector,
            "image_captioner": self.image_captioner, # NOTE Pan: please check
            "knowledge_retrieval": self.knowledge_retrieval,
            "solution_generator": self.solution_generator,
            "answer_generator": self.answer_generator,
        }

    def _get_question_text(self, example):
        context = example["hint"].strip()
        choices = example["choices"]
        inds = ["A", "B", "C", "D", "E"]
        choice_list = [f"({inds[i]}) {choices[i]}" for i in range(len(choices))]
        option = " ".join(choice_list)
        question = example["question"]

        if context != "":
            question_text = f"{question}\n\nContext: {context}\n\nOptions: {option}"
        else:
            question_text = f"{question}\n\nOptions: {option}"

        return question_text

    def _get_metadata(self, example):
        metadata = {
            "has_image": example["has_image"],
            "grade": int(example["grade"].replace("grade", "")),
            "subject": example["subject"],
            "topic": example["topic"],
            "category": example["category"],
            "skill": example["skill"],
        }
        return metadata
    
    def forward(
        self, pid: str = None, example: Dict[str, Any] = None, image_bytes: bytes = None
    ):
        question_text = self._get_question_text(example)
        metadata = self._get_metadata(example)

        self.pid = pid
        self.example = example
        self.image_bytes = image_bytes
        self.question_text = question_text
        self.metadata = metadata

        # Plan the modules
        self.predicted_modules, plan_variable = self.planner(question_text, metadata)

        # To test.
        # self.predicted_modules = [
        #    "text_detector",
        #    "knowledge_retrieval",
        #    "solution_generator",
        # ]

        # Initialize state
        self.state = {
            "solution": None,
            "tool_tape_variables": [plan_variable],
        }

        print(f"\n==> [Predicted Modules]: {self.predicted_modules}")

        # Initialize response variable
        tool_tape = tg.Variable(
            "",
            role_description="response that stores the query information, along with the execution inputs and outputs of the previous modules (tools).",
            requires_grad=True,
        )
        self.state["last_output"] = plan_variable
        self.state["tool_tape"] = tool_tape
        self.state_history = {}

        for t, module_name in enumerate(self.predicted_modules):
            if module_name not in self.module_registry:
                raise ValueError(f"Unknown module: {module_name}")

            module_func = self.module_registry[module_name]
            output = module_func(
                pid=pid,
                image_bytes=image_bytes,
                question_text=question_text,
                metadata=metadata,
                example=example,
                **self.state,
            )

            self.state["last_output"] = output

            self.state_history[t] = {
                "state": copy.deepcopy(self.state),
                "remaining_modules": self.predicted_modules[t + 1 :],
                "variable_to_optimize": output,
            }

            if ("answer" not in module_name) and ("solution" not in module_name):
                self.state["tool_tape_variables"].append(output)
                tool_tape.set_value(tool_tape.get_value() + output.get_value())
            if "solution" in module_name:
                self.state["solution"] = output

            print(f"\n==> [Module at {t}]: {module_name}")
            print(
                f"\n==> [Remaining modules at {t}]: {self.state_history[t]['remaining_modules']}"
            )
            print(
                f"\n==> [Variable to optimize at {t}]: {self.state_history[t]['variable_to_optimize']}"
            )

        self.state["final_output"] = output
        self.state["tool_tape"] = tool_tape
        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def intermediate_forward_pass(self, start_index, variable_to_optimize):
        """
        Perform an intermediate forward pass starting from a specific module.

        :param start_index: The index of the module to start from
        :param variable_to_optimize: The variable that was just optimized
        :return: The final output after running the remaining modules
        """
        remaining_modules = self.predicted_modules[start_index:]
        current_state = copy.deepcopy(self.state_history[start_index]["state"])

        # Update the state with the optimized variable in the previous step
        if start_index > 0:
            previous_module = self.predicted_modules[start_index - 1]
            if previous_module in ["text_detector", "knowledge_retrieval", "image_captioner"]: # NOTE Pan: please check
                current_state["tool_tape"].set_value(
                    current_state["tool_tape"].get_value()
                    + variable_to_optimize.get_value()
                )
            elif "solution" in previous_module:
                current_state["solution"] = variable_to_optimize

        # Generate the output after running the remaining modules
        output = variable_to_optimize
        for t, module_name in enumerate(remaining_modules, start=start_index):
            module_func = self.module_registry[module_name]
            output = module_func(
                pid=self.pid,
                image_bytes=self.image_bytes,
                question_text=self.question_text,
                metadata=self.metadata,
                example=self.example,
                **current_state,
            )

            if ("answer" not in module_name) and ("solution" not in module_name):
                current_state["tool_tape_variables"].append(output)
                current_state["tool_tape"].set_value(
                    current_state["tool_tape"].get_value() + output.get_value()
                )
            if "solution" in module_name:
                current_state["optimized_solution"] = output
                current_state["solution"] = output

            # Update state history
            self.state_history[t] = {
                "state": copy.deepcopy(current_state),
                "remaining_modules": remaining_modules[t - start_index + 1 :],
                "variable_to_optimize": output,
            }

        current_state["final_output"] = output
        return output, current_state

    def optimize(self, loss_fn: Callable):
        """
        Optimize the agent using Textual Coordinate Descent.
        Coordinate Descent is a simple optimization algorithm that optimizes one variable at a time.
        REF: https://en.wikipedia.org/wiki/Coordinate_descent

        :param loss_fn: A callable that computes the loss given the final output
        :return: The optimized final output
        """
        for t in range(len(self.predicted_modules)):
            if "text_detector" in self.predicted_modules[t]:
                continue
            if "answer_generator" not in self.predicted_modules[t]:
                print("\n")
                print("#"*100)
                print(f"==> [Optimizing module {t}]: {self.predicted_modules[t]}")
                print("#"*100)

                # Get the variable to optimize
                variable_to_optimize = self.state_history[t]["variable_to_optimize"]
                # And hopefully, currently i have the right state such that the variable_to_optimize is one of the predecessors of the final output.
                # How do I make sure that? Pull from the state history

                print(f"\033[44m\n==> [Variable to optimize at {t} (Output before optimization)]:\033[0m")
                print("="*50)
                print(variable_to_optimize.get_value())
                print("="*50)

                # Compute loss and gradients
                output = self.state["final_output"]
                loss = loss_fn(output)
                # print(f"\n==> [Loss at {t}]: {loss.get_value()}")
                loss.backward()
                # print(f"\n==> [Gradients computed]")

                # Optimize the variable
                optimizer = tg.TGD(parameters=[variable_to_optimize])
                optimizer.step()
                print(f"\033[48;5;208m\n==> [Variable outputs and gradients]:\033[0m")
                print(optimizer.gradient_memory_dict)

                print(f"\033[42m\n==> [Variable updated at {t}] (Output after optimization):\033[0m")
                print("="*50)
                print(variable_to_optimize)
                print("="*50)
    
                # Perform intermediate forward pass
                output, current_state = self.intermediate_forward_pass(
                    t, variable_to_optimize
                )

                self.state = current_state
            else:
                continue
            print(f"\n==> [Intermediate forward pass completed]")

            print("\n==> [Optimization complete]")
            print(f"\n==> [Final output]: {output.get_value()}")

        return output
