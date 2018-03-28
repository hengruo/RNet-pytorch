# ZZXproj

## Usage

Python 3.5/3.6 & PyTorch 0.3.1

1. Install pytorch 0.3.1 for Python 3.5+
2. Run `sudo pip3 -r requirements.txt`
3. Run `python3 main.py`

## Structure
config: all configurations.

data: convert raw data into word embeddings & character embeddings. (HX)

model: network model. (HZ)

trainer: trainer & tester. (MZ)

## Checkpoints
**20180226**

Implementation of R-net in PyTorch without self-attention.

[Natural Language Computing Group, MSRA: R-NET: Machine Reading Comprehension with Self-matching Networks](https://www.microsoft.com/en-us/research/publication/mrc/)


**20180326**

Requirements:

Checkpoint 2 will involve reproducing the evaluation numbers of a state-of-the-art baseline model for the task of interest with code that you have implemented from scratch (e.g., you are not allowed to simply run existing code, nor copy large chunks from an existing implementation of the particular model of interest). In other words, you must get the same numbers as the previous paper on the same dataset.
In your report, also perform an analysis of what remaining errors this model makes (ideally with concrete examples of failure cases), and describe how you plan to create a new model for the final project that will address these error cases. If you are interested in tackling a different task in the final project, you may also describe how you adopted the existing model to this new task and perform your error analysis on the new task (although you must report results on the task that the state-of-the-art model was originally designed for.

The grading rubric for this checkpoint is as follows:
* A+: Exceptional or surprising. Goes far beyond most other submissions.
* A: A complete re-implementation that meets or exceeds the state of the art. An analysis of the results, and forward-looking plans for further development.
* A-: Similarly, a complete re-implementation with competitive result numbers, but less analysis or forward-looking plans for development than assignments rewarded an A.
* B+: An implementation and evaluation numbers exist, but they do not match previous work in the field. Or the analysis or forward-looking plans may be seriously lacking.
* B or B-: Two or more of the above three elements are lacking.
* C+ or below: Clear lack of effort or incompleteness.

