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

char_emb_config = {
    "<UNK>": 0,
    "<PAD>": 1,
    "<SOS>": 2,
    "<EOS>": 3,
    "start": "<SOS>",
    "end": "<EOS>",
    "tokenization": "nltk",
    "specials": ["<UNK>", "<PAD>", "<SOS>", "<EOS>"],
    "root": char_emb_dir,
    "name": char_emb_filename,
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
    "tokenization": "nltk",
    "specials": ["<UNK>", "<PAD>", "<SOS>", "<EOS>"],
    "root": word_emb_dir,
    "name": word_emb_filename,
    "type": "glove.840B",
    "dim": 300,
    "itox": None,
    "xtoi": None,
    "vec": None
}
train_config = {
    "dir": train_dir,
    "filename": train_filename
}

char_embedding_config = {"embedding_weights": cv_vec,
                             "padding_idx": word_vocab_config["<UNK>"],
                             "update": args.update_char_embedding,
                             "bidirectional": args.bidirectional,
                             "cell_type": "gru", "output_dim": 300}

word_embedding_config = {"embedding_weights": wv_vec,
                            "padding_idx": word_vocab_config["<UNK>"],
                            "update": args.update_word_embedding}

sentence_encoding_config = {"hidden_size": args.hidden_size,
                            "num_layers": args.num_layers,
                            "bidirectional": True,
                            "dropout": args.dropout, }

pair_encoding_config = {"hidden_size": args.hidden_size,
                        "num_layers": args.num_layers,
                        "bidirectional": args.bidirectional,
                        "dropout": args.dropout,
                        "gated": True, "mode": "GRU",
                        "rnn_cell": torch.nn.GRUCell,
                        "attn_size": args.attention_size,
                        "residual": args.residual}

self_matching_config = {"hidden_size": args.hidden_size,
                        "num_layers": args.num_layers,
                        "bidirectional": args.bidirectional,
                        "dropout": args.dropout,
                        "gated": True, "mode": "GRU",
                        "rnn_cell": torch.nn.GRUCell,
                        "attn_size": args.attention_size,
                        "residual": args.residual}

pointer_config = {"hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "residual": args.residual,
                    "rnn_cell": torch.nn.GRUCell}