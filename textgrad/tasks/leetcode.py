import platformdirs
from .base import Dataset
import os
import json


class LeetCodeHardEval(Dataset):
    def __init__(self, root: str = None):
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")

        self.root = root
        data_path = f"{self.root}/leetcode-hard.jsonl"
        self._check_or_download_dataset()

        self.dataset = [json.loads(line) for line in open(data_path)]
        
        self._task_description = 'You will solve a hard coding problem from LeetCode. You will be given a prompt describing a problem. You need to write a function that passes all the tests.'

    def get_task_description(self):
        return self._task_description

    def _check_or_download_dataset(self):
        data_path = f"{self.root}/leetcode-hard.jsonl"
        if os.path.exists(data_path):
            return
        
        os.makedirs(f"{self.root}/", exist_ok=True)
        import requests
        url = "https://raw.githubusercontent.com/vinid/data/master/leetcode_with_tests.jsonl"
        r = requests.get(url)
        with open(data_path, 'wb') as f:
            f.write(r.content)

    def __getitem__(self, index):
        row = self.dataset[index]
        task_id = row["task_id"]
        prompt = row["prompt"]
        tests = row["test"]

        return task_id, prompt, tests

    def __len__(self):
        return len(self.dataset)
