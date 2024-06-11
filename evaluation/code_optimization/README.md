# Limitations of The Evals

In favor of transparency, we would like to highlight some limitations of the evaluations presented in the paper.

Getting stable results from the code optimization task is challenging. Minor changes in the code snippet can lead to different results. We have tried to mitigate this by running the optimization multiple times with different seeds and reporting the average results. 
However, the results can still vary. This is true both for TextGrad and for the baseline, [Reflexion](https://github.com/noahshinn/reflexion).

Other than noise in the results, the code optimization task is also challenging because we have to deal with pushing code snippets on the LeetCode platform. 
This can lead to timeouts, network issues, etc.  We have tried to mitigate this by trying to carefully handle the returns value and re-run where needed. 
Some of these issues are due to the rate limiting of the LeetCode platform.

Other issues are instead related to the general API and the (possibly) the package that we and Reflexion uses to wrap LeetCode APIs, namely [python-leetcode](https://github.com/fspv/python-leetcode):

For example, the API does not accept some programs if they contain unexpected characters; however some of these unexpected characters are simple parentheses or brackets wrapping 
python statements. We have tried to mitigate this by removing these characters using a call to GPT-4.

## Running the PythonEvaluator

Similarly to OpenAI's HumanEval, we are running model-generated code with this script. OpenAI reports this
disclaimer in their [HumanEval README](https://github.com/openai/human-eval).

> This program exists to run untrusted model-generated code. Users are strongly encouraged not to do so outside of a robust security sandbox. The execution call in execution.py is deliberately commented out to ensure users read this disclaimer before running code in a potentially unsafe manner. See the comment in execution.py for more information and instructions.

We suggest a similar approach for this evaluation. 
In addition to this, our entire `py_eval.py` and `utils.py` are commented and raise exceptions to prevent running code without the user actually taking
steps to edit that.

## LeetCodeHardGym and Data

Users interested in replication might also want to explore the packages used to get a better idea of
what to expect when running the interface to LeetCode:

The [LeetCodeHardGym](https://github.com/GammaTauAI/leetcode-hard-gym) people have created an amazing package, but there are a couple of things we had to
fix to change to run our procedures. We have basically taken some of their code and integrated it into our
evaluation script. Moreover, we used their code to create the dataset. LeetCodeHard data is available [here](https://github.com/vinid/data/blob/master/leetcode_with_tests.jsonl)

## Results

The results of the evaluations are available in the `results` folder (one up above this one).
We run 5 seeds for both TextGrad and reflexion.

Note: portion of this code are adapted from [Reflexion](https://github.com/noahshinn/reflexion).