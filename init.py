import os
import nltk
import requests
import zipfile
import numpy as np
import config as cf

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
    dirs = ['data/cache', 'data/embedding/char', 'data/embedding/word', 'data/squad', 'data/trained_model', 'checkpoint']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    nltk.download('punkt')
    download(cf.train_url_base, cf.train_filename, cf.train_dir)
    download(cf.dev_url_base, cf.dev_filename, cf.dev_dir)
    download(cf.char_emb_url_base, cf.char_emb_filename, cf.char_emb_dir)
    if not os.path.exists(os.path.join(cf.word_emb_dir, cf.word_emb_filename))
        download(cf.word_emb_url_base, cf.word_emb_zip, cf.word_emb_dir)
    zip_ref = zipfile.ZipFile(os.path.join(cf.word_emb_dir, cf.word_emb_zip), 'r')
    zip_ref.extractall(cf.word_emb_dir)
    zip_ref.close()
    os.remove(os.path.join(cf.word_emb_dir, cf.word_emb_zip))

def main():
    initialize()
    

if __name__ == "__main__":
    main()
    