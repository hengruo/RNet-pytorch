# ZZXproject

## Overview
[Natural Language Computing Group, MSRA: R-NET: Machine Reading Comprehension with Self-matching Networks](https://www.microsoft.com/en-us/research/publication/mrc/)

## Usage

Python 3.5/3.6 & PyTorch 0.4

1. Install pytorch 0.4 for Python 3.5+
2. Run `pip install spacy tqdm ujson requests`
3. Run `python main.py`

## Structure
dataset.py: download dataset and parse.

main.py: program entry.

models.py: R-net structure.

error_analysis.py: analyze error answers
