from textgrad.engine.openai import EngineLM

def load_multimodal_instance_task(task_name: str, evaluation_api: EngineLM, *args, **kwargs):
    if task_name == "scienceqa":
        from scienceqa import ScienceQADataset
        test_set = ScienceQADataset(evaluation_api=evaluation_api, split="test", *args, **kwargs)
        return test_set
    else:
        raise ValueError(f"Instance task {task_name} not found.")
