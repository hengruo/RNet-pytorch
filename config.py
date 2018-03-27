# char embedding layer
char_limit = 16
char_dim = 8
char_hidden_size = 100
char_num_layers = 1
char_dir = 2

# encoder layer
enc_num_layers = 3
enc_bidir = True
enc_dir = 2 if enc_bidir else 1

# global
batch_first = True
dropout = 0.7
batch_size = 64
hidden_size = 75
word_emb_size = 300
char_emb_size = char_dir * char_num_layers * char_hidden_size
emb_size = word_emb_size + char_emb_size