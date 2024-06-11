Here are the three prompts TextGrad generated for each task, using batch optimization with TextGrad:

**GSM8k:** `You will answer a mathematical reasoning question. Restate the problem in your own words to ensure understanding. Break down the problem into smaller steps, explaining each calculation in detail. Verify each step and re-check your calculations for accuracy. Use proper mathematical notation and maintain consistency with the context of the question. Always conclude with the final answer in the following format: 'Answer: $VALUE' where VALUE is a numerical value.`

<br>

**Object Counting:** `You will answer a reasoning question. List each item and its quantity in a clear and consistent format, such as '- Item: Quantity'. Sum the values directly from the list and provide a concise summation. Ensure the final answer is clearly indicated in the format: 'Answer: $VALUE' where VALUE is a numerical value. Verify the relevance of each item to the context of the query and handle potential errors or ambiguities in the input. Double-check the final count to ensure accuracy.`

<br>

**Word Sorting:** `You will answer a reasoning question. Think step by step. Clearly state the final answer in the format expected by the question. Ensure that your final answer matches the format of the expected answer, whether it is a list, a numerical value, or another type of response. When sorting or performing any ordered task, perform lexicographical comparison, which involves comparing each character in sequence until a difference is found. Clearly explain the criteria and steps involved, including how to handle ties or similar cases. For tasks involving multiple steps, break down each step in detail and explain the reasoning behind each action. Only include steps that are necessary for the task at hand, and avoid mentioning irrelevant or redundant steps. After arriving at your final answer, include a verification step that compares each element in the final output against the expected result to ensure they match. Include intermediate steps or iterations of the process to verify the accuracy of each step. Consider and address potential edge cases or assumptions, such as handling different capitalizations, special characters, or spaces, in your reasoning to demonstrate a robust understanding of the task. Ensure consistency in the presentation of information, such as using the same format for lists or other structured data throughout your response. Explicitly state the final prediction in a clear and unambiguous manner, using a formal structure such as 'Final Answer: [item1, item2, item3, ...]'. Provide a detailed explanation of your reasoning process, justifying each step, and avoid including unnecessary or incorrect information. Include a step for error handling or specify what to do if an unexpected input is encountered.`


## Notes
Since we are running prompt optimization with only a handful of examples from the training set, the training procedure has high variance. We believe there are great future research opportunities in this direction to improve stability of the optimization process.

## Experiments
Running experiments should be straightforward, below are some example commands:

```bash
python prompt_optimization.py --task=BBH_object_counting --run_validation --do_not_run_larger_model  --evaluation_engine=gpt-4o --test_engine=gpt-3.5-turbo-0125 
python prompt_optimization.py --task=BBH_word_sorting --run_validation --do_not_run_larger_model  --evaluation_engine=gpt-4o --test_engine=gpt-3.5-turbo-0125 
python prompt_optimization.py --task=GSM8K_DSPy --run_validation --do_not_run_larger_model  --evaluation_engine=gpt-4o --test_engine=gpt-3.5-turbo-0125 
```