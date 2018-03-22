batch_size = 64

train_filename = "train-v1.1.json"
dev_filename = "dev-v1.1.json"
char_emb_filename = "glove.840B.300d-char.txt"
char_emb_pickle_filename = 'char-emb.pickle'
word_emb_zip = "glove.840B.300d.zip"
word_emb_filename = "glove.840B.300d.txt"
word_emb_pickle_filename = 'word-emb.pickle'

data_dir = "data/squad"
emb_dir = "data/embedding"

train_encoded = "train.pt"
dev_encoded = "dev.pt"

word_emb_url_base = "http://nlp.stanford.edu/data/"
char_emb_url_base = "https://raw.githubusercontent.com/minimaxir/char-embeddings/master/"
train_url_base = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
dev_url_base = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"

char_emb_config = {
    "<UNK>": 0,
    "<PAD>": 1,
    "<SOS>": 2,
    "<EOS>": 3,
    "start": "<SOS>",
    "end": "<EOS>",
    "specials": ["<UNK>", "<PAD>", "<SOS>", "<EOS>"],
    "dir": emb_dir,
    "raw_name": char_emb_filename,
    "pkl_name": char_emb_pickle_filename,
    "type": "glove.840B",
    "dim": 300,
    "itox": None,
    "xtoi": None,
    "vec": None
}
word_emb_config = {
    "<UNK>": 0,
    "<PAD>": 1,
    "<SOS>": 2,
    "<EOS>": 3,
    "start": "<SOS>",
    "end": "<EOS>",
    "specials": ["<UNK>", "<PAD>", "<SOS>", "<EOS>"],
    "dir": emb_dir,
    "raw_name": word_emb_filename,
    "pkl_name": word_emb_pickle_filename,
    "type": "glove.840B",
    "dim": 300,
    "itox": None,
    "xtoi": None,
    "vec": None
}

train_config = {
    "dir": data_dir,
    "raw_name": train_filename,
    "pkl_name": train_encoded,
    "data": {
        "pword": [], # passage word embedding
        "pchar": [], # passage char embedding
        "qword": [], # question word embedding
        "qchar": [], # question char embedding
        "aword": [], # ANSWER's start word position and end word position
        "achar": [] # ANSWER's start char position and end char position
    }
}

dev_config = {
    "dir": data_dir,
    "raw_name": dev_filename,
    "pkl_name": dev_encoded,
    "data": {
        "pword": [],
        "pchar": [],
        "qword": [],
        "qchar": [],
        "aword": [],
        "achar": []
    }
}