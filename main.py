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


def initialize():
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

def train(tcfg, dcfg):
    epoch_num = 20
    train_ratio = 0.9
    t_data = tcfg['data']
    d_data = dcfg['data']
    rnet = model.RNet()
    for i in range(epoch_num):
        for t in t_data:
            pdata = t_data['context']
            qdata = t_data['answer']
            start = t_data['answer_start']
            end = t_data['answer_end']
            sword = pdata[0][start]
            eword = pdata[0][end]
            rnet.zero_grad()
            predict = rnet(pdata, qdata)
            loss = F.l1_loss(predict, torch.cat([sword, eword]))
            loss.backward()
        print('loss:\t', loss.data[0])
        correct = 0
        for d in d_data:
            pdata = d_data['context']
            qdata = d_data['answer']
            start = d_data['answer_start']
            end = d_data['answer_end']
            sword = pdata[0][start]
            eword = pdata[0][end]
            predict = rnet(pdata, qdata)
            loss = F.l1_loss(predict, torch.cat([sword, eword]))
            if loss < 0.05:
                correct += 1
        print('acc:\t', correct/len(d_data))


def main():
    initialize()
    dt.get_embedding(cf.word_emb_config)
    dt.get_embedding(cf.char_emb_config)
    dt.get_encoding(cf.train_config, cf.word_emb_config, cf.char_emb_config)
    dt.get_encoding(cf.dev_config, cf.word_emb_config, cf.char_emb_config)
    train(cf.train_config, cf.dev_config)

if __name__ == "__main__":
    main()
