# Results for Solution Optimization on Multimodal Tasks

We are pleased to release the results of our solution optimization experiments for multimodal tasks, including [MathVista](https://mathvista.github.io/) and [ScienceQA](https://scienceqa.github.io/) tasks. This release aims to ensure transparency and provide detailed insights into our optimization processes.

The files in this repository contain the solution trajectory, final solution, and the answers for each question in the datasets. Each `.json` file (can be downloaded from google drive) is structured as a dictionary where the key is the question index, and the value is another dictionary with the following possible keys:

- **`question`**: The question text.
- **`answer`**: The ground truth answer.
- **`predictions`**: The trajectory of solutions throughout the optimization process.
- **`loss_history`**: The history of loss function values over different iterations of optimization.
- **`performance_history`**: The prediction score, where 1 indicates a correct answer and 0 indicates an incorrect answer.
- **`result_data`**: The intermediate results of predictions.
- **`ques_data`**: The metadata of the question, which is useful for further fine-grained analysis.

## Notes
Solution optimization in general depends heavily on the test-time objective. Depending on the test-time objective, e.g. the model can be driven to explore more, which is the approach we took. For this reason, we used majority voting to get the final prediction. There are lots of interesting questions around identifying good test-time training strategies!

## Experiment on MathVista

[MathVista](https://mathvista.github.io/) is a comprehensive benchmark for mathematical reasoning within visual contexts. It emphasizes the diversity and complexity of visual perception and mathematical reasoning challenges.

To conduct an experiment on MathVista, use the following example command:

```sh
cd evaluation
python solution_optimization_mm.py --task mathvista \
--engine=gpt-4o \
--eval_engine=gpt-4o \
--max_iterations 4 \
--num_threads 10 \
--majority_voting
```

The resulting `mathvista_predictions.json` file can be downloaded from [here](https://drive.google.com/drive/u/0/folders/1hJ6kQozkTNiUPxtEhZCgCqnK2EFW5MHY).

## Experiment on ScienceQA

[ScienceQA](https://scienceqa.github.io/) (Science Question Answering) is a multimodal benchmark consisting of multiple-choice questions covering a diverse set of science topics. It challenges participants to understand scientific images, retrieve relevant knowledge, and provide accurate reasoning for high-school-level scientific questions.

Running an experiment on ScienceQA, use the following example command:

```sh
cd evaluation

python solution_optimization_mm.py --task scienceqa \
--engine=gpt-4o \
--eval_engine=gpt-4o \
--max_iterations 8 \
--num_threads 20
```

The resulting `scienceqa_predictions.json` file can be downloaded from [here](https://drive.google.com/file/d/1BkMD3CcaxAUpB-9L0aI8bLEqoo8jHqlZ/view?usp=sharing).
