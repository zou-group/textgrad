import os
import logging
import json
from datetime import datetime

LOG_DIR = os.getenv("TEXTGRAD_LOG_DIR", "./logs/")

__version__ = "0.1.8"

class CustomJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        super(CustomJsonFormatter, self).format(record)
        output = {k: str(v) for k, v in record.__dict__.items()}
        return json.dumps(output)

cf = CustomJsonFormatter()
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl")
sh = logging.FileHandler(log_file)
sh.setFormatter(cf)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(sh)

from .variable import Variable
from .loss import TextLoss
from .model import BlackboxLLM
from .engine import EngineLM, get_engine
from .optimizer import TextualGradientDescent, TGD
from .config import set_backward_engine, SingletonBackwardEngine
from .autograd import sum, aggregate

singleton_backward_engine = SingletonBackwardEngine()