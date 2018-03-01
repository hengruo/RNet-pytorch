batch_size = 64

train_filename = "train-v1.1.json"
dev_filename = "dev-v1.1.json"
char_emb_filename = "glove.840B.300d-char.txt"
word_emb_zip = "glove.840B.300d.zip"
word_emb_filename = "glove.840B.300d.txt"

train_dir = "data/squad"
dev_dir = "data/squad"
char_emb_dir = "data/embedding/char"
word_emb_dir = "data/embedding/word"

train_encoded = "train.pt"
dev_encoded = "dev.pt"

word_emb_url_base = "http://nlp.stanford.edu/data/"
char_emb_url_base = "https://raw.githubusercontent.com/minimaxir/char-embeddings/master/"
train_url_base = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
dev_url_base = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"

char_vocab_config = {
    "<UNK>": 0,
    "<PAD>": 1,
    "<SOS>": 2,
    "<EOS>": 3,
    "insert_start": "<SOS>",
    "insert_end": "<EOS>",
    "tokenization": "nltk",
    "specials": ["<UNK>", "<PAD>", "<SOS>", "<EOS>"],
    "embedding_root": char_emb_dir,
    "embedding_name": char_emb_filename,
    "embedding_type": "glove.840B",
    "embedding_dim": 300
}
word_vocab_config = {
    "<UNK>": 0,
    "<PAD>": 1,
    "<SOS>": 2,
    "<EOS>": 3,
    "insert_start": "<SOS>",
    "insert_end": "<EOS>",
    "tokenization": "nltk",
    "specials": ["<UNK>", "<PAD>", "<SOS>", "<EOS>"],
    "embedding_root": word_emb_dir,
    "embedding_name": word_emb_filename,
    "embedding_type": "glove.840B",
    "embedding_dim": 300
}
train_config = {
    "dir": train_dir,
    "filename": train_filename
}