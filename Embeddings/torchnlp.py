import builtins
import torch
import torchtext
import collections
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = None
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

def load_dataset(ngrams=1,min_freq=1):
    global vocab, tokenizer
    print("Loading dataset...")
    train_dataset, test_datset = torchtext.datasets.AG_NEWS(root='./data')
    train_dataset = list(train_dataset)
    test_datset = list(test_datset)
    classes = ['world', 'sports', 'Business', 'Sci/Tech']
    print('Building Vocab...')
    counter = collections.Counter()
    for (label, line) in train_dataset:
        counter.update(torchtext.data.utils.ngrams_iterator(tokenizer(line),ngrams=ngrams))
    vocab = torchtext.vocab.vocab(counter, min_freq=min_freq)
    return train_dataset, test_datset, classes, vocab

stoi_hash = {}
def encode(x,voc=None,unk=0,tokenizer=tokenizer):
    global stoi_hash
    v = vocab if voc is None else voc
    if v in stoi_hash.keys():
        stoi = stoi_hash[v]
    else:
        stoi = v.get_stoi()
        stoi_hash[v]=stoi
    return [stoi.get(s,unk)for s in tokenizer(x)]