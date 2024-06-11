# Results for solution optimization

Here we release the results of the solution optimization experiments for GPQA and MMLU subsets, for the sake of transparency.

Files in this folder contain the solution trajectory, final solution, and the answer for each question in these datasets. In particular, each .json file is a dictionary where the key is the question, and the value is a dictionary with two keys: "predictions" (the trajectory of solutions, and the last item is the final majority-voting prediction) and "answer" (ground truth answer).

## Notes
Solution optimization in general depends heavily on the test-time objective. Depending on the test-time objective, e.g. the model can be driven to explore more, which is the approach we took. For this reason, we used majority voting to get the final prediction. There are lots of interesting questions around identifying good test-time training strategies!