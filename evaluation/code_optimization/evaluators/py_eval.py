# from dotenv import load_dotenv
# from utils import get_output_of_test, timeout_exec
#
# class PythonEvaluator:
#     """
#     This is a simple evaluator for Python code.
#     """
#
#     def __init__(self):
#         raise Exception("You should remove this and remove the comments, but run at your own risk!!!"
#                         "This is going to run model-generated code on your machine.")
#
#     def evaluate(self, code, tests):
#         """
#         This function evaluates the code with the given tests.
#         Returns a string that contains the tests that passed and failed.
#
#
#         :param code:
#         :param tests:
#         :return:
#         """
#         raise Exception("You should remove this and remove the comments, but run at your own risk!!!"
#                         "This is going to run model-generated code on your machine.")
#         func_test_list = [f'from typing import *\n{code}\n\n{test}' for test in tests]
#
#         success_tests = []
#         failed_tests = []
#         num_tests = len(func_test_list)
#         for i in range(num_tests):
#
#             try:
#                 #timeout_exec(func_test_list[i])
#                 success_tests += [tests[i]]
#             except Exception as e:
#                 failed_test = tests[i]
#
#                 try:
#                     #output = get_output_of_test(code, failed_test)
#                     asserted_value = tests[i].split("==")[1].strip()
#                     failed_tests += [f"{tests[i]} # ERROR: This unit test fails. Output was {output}, but expected value was: {asserted_value}"]
#                 except Exception as e:
#                     failed_tests += [f"{tests[i]} # ERROR: This unit test fails because the function generated: {e}."]
#
#         state = []
#         for test in tests:
#             if test in success_tests:
#                 state += [True]
#             else:
#                 state += [False]
#
#         state = tuple(state)
#
#         feedback = "**Tests that the code passed:**\n"
#         if len(success_tests) == 0:
#             feedback += "\nNo tests passed.\n"
#         else:
#             for test in success_tests:
#                 feedback += f"\n{test}"
#         feedback += "\n\n**Tests that the code failed:**\n"
#         if len(failed_tests) == 0:
#             feedback += "\nNo tests failed.\n"
#         else:
#             for test in failed_tests:
#                 feedback += f"\n{test}"
#
#         return state, feedback