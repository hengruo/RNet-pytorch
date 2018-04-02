import json
import nltk

nltk.download('perluniprops')
nltk.download('nonbreaking_prefixes')

from word_model import Vocab ,WordModel
import pickle
import coloredlogs, logging
from encoder import Encoder
from torch.autograd import Variable
import torch
import torch.nn as nn


# Create a logger object.
logger = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG')
coloredlogs.install(fmt='%(asctime)s,%(msecs)03d %(levelname)s %(message)s')

dataset = json.load(open('data/dev-v1.1.json'))

word_model = WordModel()
logger.warning('Generating Vocab ...')
word_model.store_into_vocab(dataset)
