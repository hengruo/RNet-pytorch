import os
import requests
import zipfile
import numpy as np
import config as cf
import data as dt
import model
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config as cfg

def download(urlbase, filename, path):
    url = os.path.join(urlbase, filename)
    if not os.path.exists(os.path.join(path, filename)):
        try:
            print("Downloading file {}...".format(filename))
            r = requests.get(url, stream=True)
            fullname = os.path.join(path, filename)
            with open(fullname, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        except AttributeError as e:
            print("Download error!")
            raise e


def data_prepare():
    '''
    download squad dataset into `data/squad` and embedding into `data/embedding`
    '''
    dirs = [cf.data_dir, cf.emb_dir]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    download(cf.train_url_base, cf.train_filename, cf.data_dir)
    download(cf.dev_url_base, cf.dev_filename, cf.data_dir)
    download(cf.char_emb_url_base, cf.char_emb_filename, cf.emb_dir)
    if not os.path.exists(os.path.join(cf.emb_dir, cf.word_emb_filename)):
        download(cf.word_emb_url_base, cf.word_emb_zip, cf.emb_dir)
    zip_ref = zipfile.ZipFile(os.path.join(
        cf.emb_dir, cf.word_emb_zip), 'r')
    zip_ref.extractall(cf.emb_dir)
    zip_ref.close()
    os.remove(os.path.join(cf.emb_dir, cf.word_emb_zip))

def get_embedding(emb_config):
    '''
    read embedding files by the configuration of embeddings.
    input: 
        embedding configuration (cfg.char_emb_config or cfg.word_emb_config)
    return: 
        None. 
        But it will change 3 fields in the configuration of embeddings:
            vec: embedding list
            itox: index to word/char. e.g. itox(0) = "<UNK>".
            xtoi: word/char to index. e.g. xtoi("<UNK>") = 0.
    '''
    pass

def get_encoding(data_config):
    '''
    read dataset and convert it into a sequence of indexes of embeddings.
    input: 
        data configuration (cfg.train_config or cfg.dev_config)
    return: 
        None. 
        But it will add encoded data into data configuration.
    '''
    pass