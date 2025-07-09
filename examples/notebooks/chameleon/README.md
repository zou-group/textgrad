

## TextGrad Optimization for the Agentic System Chameleon

Here, we provide the notebook examples for the agentic system [Chameleon](https://arxiv.org/abs/2304.09842).

Chameleon is a complex AI system that can plan, execute, and reason about a sequence of tools to answer a question. TextGrad is used to optimize the long chain of reasoning steps in Chameleon in an interactive self-improvement loop.

One example is provided: [TextGrad_Chemeleon_ScienceQA.ipynb](TextGrad_Chemeleon_ScienceQA.ipynb).

## Install the required packages

Install the required packages using the following commands:

```sh
conda create --name textgrad
conda activate textgrad
cd textgrad
pip install -e .
pip install matplotlib
pip install easyocr
```

Also install:
```sh
sudo apt install graphviz
```

For more installation instructions, please refer to the [installation instructions](https://github.com/zou-group/textgrad?tab=readme-ov-file#installation) in the [textgrad](https://github.com/zou-group/textgrad) repository. 
