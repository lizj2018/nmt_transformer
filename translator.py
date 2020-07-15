# _*_ coding: UTF-8 _*_

import torch
import os
import time
from os.path  import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
import numpy as np  
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
import random
import pandas as pd
#from torch.autograd import Variable

from torchtext import data, datasets
import spacy
from tqdm import tqdm

'''
pip install revtok
pip install torchtext spacy
python -m spacy download en
python -m spacy download de



'''

device= ("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

    

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def forward(self,src,tgt,src_mask,tgt_mask):
        return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src),src_mask)
    def decode(self,memory,src_mask,tgt,tgt_mask):
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)
    
class Generator(nn.Module):
    def __init__(self,d_model, vocab):
        super(Generator,self).__init__()
        self.proj= nn.Linear(d_model,vocab)
    
    def forward(self,x):
        return F.log_softmax(self.proj(x), dim=-1)

    

curdir=os.path.abspath(os.curdir)
corpus = open(curdir+'/%s-%s.txt'%('eng','ita'),'r',encoding='utf-8').readlines()
#print(len(corpus))
random.shuffle(corpus)

def prepare_data(lang1_name,lang2_name,reverse=False):
    print("Reading lines...")
    input_lang, output_lang = [],[] # raw data
    for parallel in corpus:
        so, ta =parallel[:-1].split('\t')  #separated by tab
        if so.strip() =="" or ta.strip() =="":
            continue
        input_lang.append(so)
        output_lang.append(ta)
    if reverse:
        return output_lang,input_lang
    else:
        return input_lang,output_lang

input_lang, output_lang = prepare_data("eng",'ita',True)

# get train_list, valid_list, test_list repectively

train_list=corpus[:-3756]
valid_list=corpus[-1622:]
test_list=corpus[-3756:-1622]
c_train={'src':input_lang[:-3756],'trg':output_lang[:-3756]}
train_df=pd.DataFrame(c_train)
c_valid={'src':input_lang[-1622:],'trg':output_lang[-1622:]}
valid_df=pd.DataFrame(c_valid)
c_test ={'src':input_lang[-3756:-1622], 'trg':output_lang[-3756:-1622]}
test_df=pd.DataFrame(c_test)




#counstruct dataset
# 
# 

spacy_it = spacy.load('it_core_news_sm')  
spacy_en = spacy.load('en_core_web_sm')

def tokenize_it(text):
    return [tok.text for tok in spacy_it.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]



#define configuration info of fields

BOS_WORD= '<s>'
EOS_WORD='</s>'
BLANK_WORD='<blank>'

SRC=data.Field(tokenize=tokenize_it, pad_token=BLANK_WORD)
TGT= data.Field(tokenize=tokenize_en, init_token=BOS_WORD, eos_token =EOS_WORD, pad_token =BLANK_WORD)

# construct dataset
def get_dataset( csv_data, text_field, label_field, test=False):
    fields= [('id',None),('src', text_field), ('trg', label_field)]
    examples=[]
    if test:
        for text in tqdm(csv_data['src']):
            examples.append(data.Example.fromlist([None,text,None], fields))
    
    else:
        for text, label in tqdm(zip(csv_data['src'],csv_data['trg'])):
            examples.append(data.Example.fromlist([None,text,label],fields))
    
    return examples,fields


train_examples, train_fields = get_dataset(train_df, SRC,TGT)
valid_examples, valid_fields = get_dataset(valid_df,SRC,TGT)
test_examples, test_fields = get_dataset(test_df,SRC,None,True)

# construct datasets
#vars() 函数返回对象object的属性和属性值的字典对象。
MAX_LEN=100
train= data.Dataset(train_examples, train_fields,
        filter_pred=lambda x: len(vars(x)['src']) <=MAX_LEN and len(vars(x)['trg']) <=MAX_LEN)
valid= data.Dataset(valid_examples, valid_fields,
        filter_pred=lambda x: len(vars(x)['src']) <=MAX_LEN and len(vars(x)['trg']) <=MAX_LEN)
test= data.Dataset(test_examples, test_fields,
        filter_pred=lambda x: len(vars(x)['src']) <=MAX_LEN )



#construct the dictionary

MIN_FREQ=2 
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d,random_shuffle):
                for p in data.batch(d,self.batch_size *100):
                    p_batch=data.batch(
                        sorted(p,key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                
                for b in random_shuffle(list(p_batch)):
                    yield b
            self.batches = pool(self.data(),self.random_shuffler)
        else:
            self.batches =[]
            for b in data.batch(self.data(),self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b,key=self.sort_key))


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count,sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count ==1 :
        max_src_in_batch=0
        max_tgt_in_batch=0
    
    max_src_in_batch = max(max_src_in_batch,len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,len(new.trg)+2)
    src_elements =count * max_src_in_batch
    tgt_elements =count * max_tgt_in_batch
    return max(src_elements,tgt_elements)

BATCH_SIZE=1000
train_iter = MyIterator(train,batch_size=BATCH_SIZE, device=0,repeat=False,
                        sort_key =lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True)

valid_iter = MyIterator(valid,batch_size=BATCH_SIZE, device=0,repeat=False,
                        sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=False)




class BatchMask:
    def __init__(self, src, trg=None, pad=0):
        self.src=src
        self.src_mask= (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:,:-1]
            self.trg_y= trg[:,1:]
            self.trg_mask = self.make_std_mask(self.trg,pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    

    @staticmethod
    def make_std_mask(tgt,pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return  tgt_mask


def subsequent_mask(size):
    attn_shape = (1,size,size)
    subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0


def batch_mask(pad_idx,batch):
    src,trg = batch.src.transpose(0,1), batch.trg.transpose(0,1)
    return BatchMask(src,trg,pad_idx)

pad_idx=TGT.vocab.stoi["<blank"]


class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        super(Embeddings,self).__init__()
        self.lut = nn.Embedding(vocab,d_model)
        self.d_model =d_model
    def forward(self,x):
        return self.lut(x)* math.sqrt(self.d_model)



class PositionalEncoding(nn.Module):
    def __init__(self,d_model,droput,max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=droput)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0,max_len).unsqueeze(1).float()
        div_term= torch.exp(torch.arange(0,d_model,2).float() * -(math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position* div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    
    def forward(self,x):
        y=self.pe[:,:x.size(1)].requires_grad_(requires_grad = True)
        x=x+ y
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a_2=nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps=eps
    
    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std  = x.std(-1,keepdim=True)
        return self.a_2 * (x-mean)/(std +self.eps) + self.b_2



def attention(query,key,value,mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask ==0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    context = torch.matmul(p_attn,value)
    return context,p_attn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):
        super(MultiHeadedAttention,self).__init__()
        assert d_model % h == 0
        self.d_k =d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model,d_model),4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self,query,key,value,mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)  
        query,key,value = [ l(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2)
                            for l,x in zip(self.linears,(query,key,value))]
        x, self.attn = attention (query,key,value,mask=mask, dropout=self.dropout)
        x = x.transpose(1,2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)




class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1=nn.Linear(d_model,d_ff)
        self.w_2=nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SubLayerConnection(nn.Module):
    def __init__(self, size,dropout):
        super(SubLayerConnection,self).__init__()
        self.norm = LayerNorm(size)
        self.dropout= nn.Dropout(dropout)

    def forward(self,x,sublayer):
        norm_x = self.norm(x)
        sub_x = sublayer(norm_x)
        sub_x = self.dropout(sub_x)
        return x + sub_x

class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn=self_attn
        self.feed_forward =feed_forward
        self.residual_conn = clones(SubLayerConnection(size,dropout),2)
        self.size=size
    def forward(self,x,mask):
        x=self.residual_conn[0](x,lambda x:self.self_attn(x,x,x,mask))
        return self.residual_conn[1](x,self.feed_forward)
    
class Encoder(nn.Module):
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layers =clones(layer,N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, mask):
        for layer in self.layers:
            x =layer(x,mask)
        return self.norm(x)




class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        super(DecoderLayer,self).__init__()
        self.size =size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.residual_conn = clones(SubLayerConnection(size,dropout),3)
    def forward(self,x, memory, src_mask, tgt_mask):
        m =memory
        x=self.residual_conn[0](x,lambda x: self.self_attn(x,x,x,tgt_mask))
        x=self.residual_conn[1](x,lambda x: self.src_attn(x,m,m,src_mask))
        return self.residual_conn[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x =layer(x,memory,src_mask, tgt_mask)
        
        return self.norm(x)

class LabelSmoothing(nn.Module):
    def __init__(self,size, padding_idx,  smoothing=0.0):
        super(LabelSmoothing,self).__init__()
        self.criterion = nn.KLDivLoss(size_average = False)
        self.padding_indx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
    def forward (self,x,target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing /(self.size-2))
        true_dist.scatter_(1,target.data.unsqueeze(1), self.confidence)
        true_dist[:,self.padding_indx]=0
        mask = torch.nonzero(target.data == self.padding_indx)
        if mask.dim() >0:
            true_dist.index_fill_(0, mask.squeeze(),0.0)
        self.true_dist = true_dist

        y=true_dist.requires_grad_(requires_grad = True)
        return self.criterion(x, y)


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step =0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate =0
    
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()
    
    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))


class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt =opt
    
    def __call__ (self,x,y,norm):
        x= self.generator(x)
        loss = self.criterion(x.contiguous().view(-1,x.size(-1)), y.contiguous().view(-1))/norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm 

 # N: the number of layer of Encoder or Decoder
# d_ff  the dimension of feedforward neural networks.
def make_model(src_vocab, tgt_vocab, N=6, d_model=512,d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn= MultiHeadedAttention(h,d_model)
    ff = PositionwiseFeedForward(d_model,d_ff,dropout)
    position = PositionalEncoding(d_model,dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model,c(attn), c(ff),dropout), N),
        Decoder(DecoderLayer(d_model,c(attn), c(attn),c(ff),dropout), N ),
        nn.Sequential(Embeddings(d_model,src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model,tgt_vocab), c(position)),
        Generator(d_model,tgt_vocab))

    for p in model.parameters():
        if p.dim() >1:
            nn.init.xavier_uniform_(p)
    return model


USE_CUDA= torch.cuda.is_available()
print_every = 50
plot_every =100
plot_losses =[]
def time_since(t):
    now = time.time()
    s = now -t
    m = math.floor(s/90)
    s -= m*60
    return '%dm %ds ' % (m,s)

def run_epoch(data_iter, model, loss_compute):
    start_epoch = time.time()
    total_tokens = 0
    total_loss =0
    tokens = 0
    plot_loss_total =0
    plot_tokens_total =0

    for i, batch in enumerate(data_iter):
        src = batch.src.cuda() if USE_CUDA else batch.src
        trg = batch.trg.cuda() if USE_CUDA else batch.trg
        src_mask = batch.src_mask.cuda() if USE_CUDA else batch.src_mask
        trg_mask = batch.trg_mask.cuda() if USE_CUDA else batch.trg_mask
        model =model.cuda() if USE_CUDA else model
        out= model.forward(src, trg,src_mask, trg_mask)
   
        trg_y = batch.trg_y.cuda() if USE_CUDA else batch.trg_y
        ntokens = batch.ntokens.cuda() if USE_CUDA else batch.ntokens
        
        loss=loss_compute(out,trg_y,ntokens)
        total_loss +=loss
        plot_loss_total +=loss
        total_tokens += ntokens
        plot_tokens_total += ntokens
        tokens += ntokens
        
        if i % print_every  ==1:
            elapsed = time.time() - start_epoch
            print("Epoch Step: %ed Loss: %10f time:%8s Toekns per sec: %6.0f Step: %6d Lr: %0.8f" %
                (i, loss/ntokens, time_since(start),tokens /elapsed,
                loss_compute.opt._step if loss_compute.opt is not None else 0,
                loss_compute.opt._rate if loss_compute.opt is not None else 0 ))
            tokens =0
            start_epoch = time.time()
        if i% plot_every ==1:
            plot_loss_avg = plot_loss_total / plot_tokens_total
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            plot_tokens_total =0
    return total_loss /total_tokens



model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
criterion = LabelSmoothing(size =len(TGT.vocab), padding_idx=pad_idx, smoothing =0.1)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 8000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))


start=time.time()
for epoch in range(20):
    print("EPOCH",epoch,'-----------------------------------------------------------')
    model.train()
    run_epoch((batch_mask(pad_idx,b) for b in train_iter),
                model,
                SimpleLossCompute(model.generator, criterion, opt=model_opt))
    model.eval()
    loss=run_epoch((batch_mask(pad_idx, b) for b in valid_iter),
            model,
            SimpleLossCompute(model.generator, criterion, opt=None))
    print(loss)





## save model

state ={'model': model.state_dict(), 'optimizer': model_opt, 'epoch': epoch, 'loss': loss, 'plot_losses': plot_losses}
torch.save(state, 'mt_transformer_it&en%02d.pth.tar'%(epoch))







    













