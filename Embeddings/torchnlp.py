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

def train_epoch(net, dataloader, lr=0.01,optimizer=None,loss_fn=torch.nn.CrossEntropyLoss(),epoch_size=None, report_freq=200):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = loss_fn.to(device)
    net.train()
    total_loss,acc,count,i = 0,0,0,0
    for labels, features in dataloader:
        optimizer.zero_grad()
        features, labels = features.to(device), labels.to(device)
        out = net(features)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss
        _,predicted = torch.max(out,1)
        acc +=(predicted==labels).sum()
        count+=len(labels)
        i+=1
        if i%report_freq==0:
             print(f"{count}: acc={acc.item()/count}")
        if epoch_size and count>epoch_size:
            break
    return total_loss.item()/count, acc.item()/count

def offsetify(b):
     x = [torch.tensor(encode(t[1])) for t in b]
     o = [0] + [len(t) for t in x]
     o = torch.tensor(o[:-1]).cumsum(dim=0)
     return( torch.LongTensor(o[:-1] for t in b), torch.cat(x), o)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, collate_fn=offsetify, shuffle=True)