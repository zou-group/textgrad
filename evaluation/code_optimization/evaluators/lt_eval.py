import hashlib
import time

try:
    import leetcode
    import leetcode.auth
    from leetcode_env.types import LeetCodeSubmission, ProgrammingLanguage
except:
    raise Exception("Please install leetcodehard gym (https://github.com/GammaTauAI/leetcode-hard-gym)")

from textgrad.engine import get_engine
from diskcache import Cache
import os
from dotenv import load_dotenv

# if black is missing generate exception

try:
    from black import format_str, FileMode
    import autopep8
except:
    raise Exception("Please install black and autopep8")

import ast


class LeetCodeEvaluator:
    """
    Wrapper on top of the leetcode hard gym package.
    Note that this needs an environment variable, namely LEETCODE_SESSION to be set.
    In addition to this, the leetcode-cli package has to be manually edited:

    See the following issue on GitHub:
    https://github.com/GammaTauAI/leetcode-hard-gym/issues/20

    :return:
    """

    def __init__(self):
        configuration = leetcode.Configuration()
        leetcode_session = os.environ["LEETCODE_SESSION"]
        csrf_token = os.environ["LEETCODE_CSRF_TOKEN"]

        configuration.api_key["x-csrftoken"] = csrf_token
        configuration.api_key["csrftoken"] = csrf_token
        configuration.api_key["LEETCODE_SESSION"] = leetcode_session
        configuration.api_key["Referer"] = "https://leetcode.com"
        configuration.debug = False
        self.leet_code_cache = Cache("./cache_leetcode")

        self.api_instance = leetcode.DefaultApi(leetcode.ApiClient(configuration))

    def id_from_slug(self, slug: str) -> str:
        """
        Retrieves the id of the question with the given slug
        """
        graphql_request = leetcode.GraphqlQuery(
            query="""
                      query getQuestionDetail($titleSlug: String!) {
                        question(titleSlug: $titleSlug) {
                          questionId
                        }
                      }
                  """,
            variables={"titleSlug": slug},
            operation_name="getQuestionDetail",
        )
        response = ast.literal_eval(str(self.api_instance.graphql_post(body=graphql_request)))
        frontend_id = response['data']['question']['question_id']
        return frontend_id

    def submit_for_evaluation(self, code, problem_tag):
        from leetcode_env.utils.formatting import PythonSubmissionFormatter

        # need to go from humaneval to leetcode and then rsstrip newlines
        code = PythonSubmissionFormatter.to_leetcode(code).rstrip().lstrip()

        # format the code
        code = format_str(code, mode=FileMode())
        code = autopep8.fix_code(code)

        # Some programs are not well formatted for the leetcode API
        TEST_ENGINE = "gpt-4o"
        ENGINE_API = get_engine(engine_name=TEST_ENGINE)
        FIXING_CODE_PROMPT = """Remove the unnecessary parentheses from the code. DO NOT CHANGE THE CODE.
Use a Python code block to write your response. For example:
```python
print('Hello world!')
```
"""
        code = ENGINE_API.generate(code, system_prompt=FIXING_CODE_PROMPT)
        code = code.split("```python")[1].split("```")[0]

        sub = LeetCodeSubmission(code=code,
                                 lang=ProgrammingLanguage.PYTHON3,
                                 question_slug=problem_tag,
                                 timeout=10)

        sub.question_id = self.id_from_slug(problem_tag)

        submission = leetcode.Submission(
            judge_type="large",
            typed_code=code,
            question_id=sub.question_id,
            test_mode=False,
            lang=sub.lang.value,
        )

        try:

            # we submit a program and then we wait for the submission to be processed
            submission_id = self.api_instance.problems_problem_submit_post(
                problem=problem_tag, body=submission)
            time.sleep(10)

        except Exception as e:
            print("Error in submission", problem_tag)
            print(e)
            print("***" * 10)
            print(code)
            print("***" * 10)
            return {"status_msg": False, "total_correct": 0, "total_testcases": 0}

        time.sleep(5)

        submission_result = self.api_instance.submissions_detail_id_check_get(
            id=submission_id.submission_id
        )

        return submission_result

    def check_if_in_cache_or_submit(self, task_id, code):
        key = hashlib.sha256(f"{task_id}__{code}".encode()).hexdigest()
        key_run_success = f"{key}_run_success"
        key_total_correct = f"{key}_total_correct"
        key_total_tests = f"{key}_total_tests"
        key_runtime = f"{key}_runtime"

        if key_run_success in self.leet_code_cache:
            success_or_error = self.leet_code_cache[key_run_success]
            total_correct = self.leet_code_cache[key_total_correct]
            total_tests = self.leet_code_cache[key_total_tests]
            runtime_res = self.leet_code_cache[key_runtime]
            print(f"Task ID: {task_id} - Success: {success_or_error} - Correct: {total_correct}/{total_tests}")

        else:
            response = self.submit_for_evaluation(code, task_id)

            if "status_msg" not in response:
                print(response)
                self.leet_code_cache[key_run_success] = False
                self.leet_code_cache[key_total_correct] = 0
                self.leet_code_cache[key_total_tests] = 0
                self.leet_code_cache[key_runtime] = -1
                return False, 0, 0, -1

            if response["status_msg"] == "Accepted":
                success_or_error = True
            else:
                success_or_error = False

            total_correct = response["total_correct"]
            total_tests = response["total_testcases"]
            runtime_res = response["status_runtime"]

            self.leet_code_cache[key_run_success] = success_or_error
            self.leet_code_cache[key_total_correct] = total_correct
            self.leet_code_cache[key_total_tests] = total_tests
            self.leet_code_cache[key_runtime] = runtime_res

            print(f"Task ID: {task_id} - Success: {success_or_error} - Correct: {total_correct}/{total_tests}")
        return success_or_error, total_correct, total_tests, runtime_res

