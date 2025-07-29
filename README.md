# Zoe LLM

> [!WARNING]
> **This project is in its experimental phase.**

A program for training and testing LLM models written in Python, with a beginner-friendly structure.

--
## Table of Contents

1. [Description](#description)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Testing](#testi)
---

## Description

Zoe LLM is a simple and modular toolkit for:

- Downloading and preprocessing datasets
- Building, training, and testing GPT-like models from scratch
- Support for verbose logging (DEBUG=1) or progress bar (tqdm).
- Interactive chat pipeline for testing
---

## Requirements

### Minimum Requirements

- CPU with bfloat16 support and six cores (12 threads)
- Memory: ≥8GB RAM.
- Disk: ≥32GB free for datasets and checkpoints.
- Python: 3.11

### Recommended Requirements

- CPU with bfloat16 support and twelve cores (24 threads)
- SSD for fast I/O.
- 32GB RAM (depending on model size).

# Installation
To install the necessary programs and dependencies, we must clone the repository on the machine that will be used to build the model and set up a virtual environment with the necessary dependencies (remembering that a Python 3.11 interpreter must be installed on the system).

``bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt``



# Configuration
First, we must download the datasets to build the model by running python utils/download_datasets.py
> You can also import other datasets manually or modify the script to include more datasets ;)

After downloading the datasets for the model, we must combine them and process them before running the tokenizer. To do this, simply run python utils/preprocess.py and to build the tokenizer, simply run the following scripts:


``python utils/compile_unigram.py
python utils/tokenizer.py``



**I RECOMMEND TESTING THE TOKENIZER USING OUR TESTING TOOL python utils/test_tokenizer.py**

After completing the necessary steps for the tokenizer, you can now train your model using python utils/train.py.
Adjust the parameters in the train.py, model.py, and chat.py files as needed.

> ⏳ **Model training can take a few hours depending on the situation...**

# Testing
To test the model, you can use python utils/chat.py to access the chat program that will interact with the model.

reescreve o Markdown sem erros de formatação
