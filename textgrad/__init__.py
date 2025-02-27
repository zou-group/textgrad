import os
import logging
import json
from datetime import datetime

from .i18n import _set_language

# Set the language of strings fed to LLMs according to the following environment variables:
# LANGUAGE, LC_ALL, LC_MESSAGES, LANG.
# If none of them is found, it fallbacks to the original string literals as is.
_set_language()


class CustomJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        super(CustomJsonFormatter, self).format(record)
        output = {k: str(v) for k, v in record.__dict__.items()}
        return json.dumps(output, indent=4)

cf = CustomJsonFormatter()
os.makedirs("./logs/", exist_ok=True)
sh = logging.FileHandler(f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl")
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
