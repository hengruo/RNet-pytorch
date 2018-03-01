import os
import pickle

import torch

from train import Trainer
from utils.utils import prepare_data, get_args, read_embedding
import config


def read_vocab(vocab_config):
    """
    :param counter: counter of words in dataset
    :param vocab_config: word_embedding config: (root, word_type, dim)
    :return: itos, stoi, vectors
    """
    wv_dict, wv_vectors, wv_size = read_embedding(vocab_config["embedding_root"],
                                                  vocab_config["embedding_type"],
                                                  vocab_config["embedding_dim"])

    # embedding size = glove vector size
    embed_size = wv_vectors.size(1)
    print("word embedding size: %d" % embed_size)

    itos = vocab_config['specials'][:]
    stoi = {}

    itos.extend(list(w for w, i in sorted(wv_dict.items(), key=lambda x: x[1])))

    for idx, word in enumerate(itos):
        stoi[word] = idx

    vectors = torch.zeros([len(itos), embed_size])

    for word, idx in stoi.items():
        if word not in wv_dict or word in vocab_config['specials']:
            continue
        vectors[idx, :wv_size].copy_(wv_vectors[wv_dict[word]])
    return itos, stoi, vectors


def main():
    args = get_args()
    itos, stoi, wv_vec = read_vocab(config.word_vocab_config)
    itoc, ctoi, cv_vec = read_vocab(config.char_vocab_config)

    train_cache = "./data/cache/SQuAD%s.pkl" % ("_debug" if args.debug else "")
    dev_cache = "./data/cache/SQuAD_dev%s.pkl" % ("_debug" if args.debug else "")

    train_json = args.train_json
    dev_json = args.dev_json

    train = read_dataset(train_json, itos, stoi, itoc, ctoi, train_cache, args.debug)
    dev = read_dataset(dev_json, itos, stoi, itoc, ctoi, dev_cache, args.debug, split="dev")

    dev_dataloader = dev.get_dataloader(args.batch_size_dev)
    train_dataloader = train.get_dataloader(args.batch_size, shuffle=True, pin_memory=args.pin_memory)

    trainer = Trainer(args, train_dataloader, dev_dataloader,
                      config.char_embedding_config, config.word_embedding_config,
                      config.sentence_encoding_config, config.pair_encoding_config,
                      config.self_matching_config, config.pointer_config)
    trainer.train(args.epoch_num)


def read_dataset(json_file, itos, stoi, itoc, ctoi, cache_file, is_debug=False, split="train"):
    if os.path.isfile(cache_file):
        print("Read built %s dataset from %s" % (split, cache_file), flush=True)
        dataset = pickle.load(open(cache_file, "rb"))
        print("Finished reading %s dataset from %s" % (split, cache_file), flush=True)

    else:
        print("building %s dataset" % split, flush=True)
        from utils.dataset import SQuAD
        dataset = SQuAD(json_file, itos, stoi, itoc, ctoi, debug_mode=is_debug, split=split)
        pickle.dump(dataset, open(cache_file, "wb"))
    return dataset


if __name__ == "__main__":
    main()