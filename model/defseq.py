# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from model.charcnn import CharCNN
from model.hypernym import Hypernym


class DefSeq(nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_dim,
                 hid_dim,
                 device,
                 pretrain_emb=None,
                 dropout=0,
                 use_i=False,
                 use_h=False,
                 use_g=True,
                 use_ch=True,
                 use_he=False,
                 **kwargs):
        super(DefSeq, self).__init__()

        self.device = device
        self.use_i = use_i
        self.use_h = use_h
        self.use_g = use_g
        self.use_ch = use_ch
        self.use_he = use_he

        char_emb_dim = 0
        char_hid_dim = 0
        char_len = 0
        he_dim = 0

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        if pretrain_emb is not None:
            # self.embedding.weight.data.copy_(pretrain_emb)
            self.embedding.from_pretrained(pretrain_emb, freeze=True)
            # self.embedding.weight.requires_grad = False

        if self.use_ch:
            print("build char sequence feature extractor: CNN ...")
            char_vocab_size = kwargs['char_vocab_size']
            char_emb_dim = kwargs['char_emb_dim']
            char_hid_dim = kwargs['char_hid_dim']
            char_len = kwargs['char_len']
            self.ch = CharCNN(char_vocab_size, None, char_emb_dim,
                              char_hid_dim, dropout, device)
        if self.use_he:
            print("build Hypernym Embeddings...")
            he_dim = emb_dim
            self.he = Hypernym(emb_dim, self.embedding, device)

        final_word_dim = emb_dim + char_hid_dim * char_len + he_dim
        self.word_linear = nn.Linear(final_word_dim, hid_dim)
        self.s_lstm = nn.LSTMCell(emb_dim, hid_dim)

        if self.use_i:
            self.i_lstm = nn.LSTMCell(final_word_dim + emb_dim, hid_dim)
        if self.use_h:
            self.h_linear = nn.Linear(final_word_dim + hid_dim, hid_dim)
        if self.use_g:
            self.g_zt_linear = nn.Linear(final_word_dim + hid_dim, hid_dim)
            self.g_rt_linear = nn.Linear(final_word_dim + hid_dim,
                                         final_word_dim)
            self.g_ht_linear = nn.Linear(final_word_dim + hid_dim, hid_dim)

        self.hidden2tag_linear = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs, init_hidden=None):
        word = inputs['word']
        seq = inputs['seq']
        chars = inputs['chars']
        hnym = inputs['hnym']
        hnym_weights = inputs['hnym_weights']

        batch_size = word.size(0)
        seq_size = seq.size(0)

        word_embeddings = self.embedding(word)
        seq_embeddings = self.embedding(seq)
        if self.use_ch:
            char_embeddings = self.ch(chars)
            word_embeddings = torch.cat(
                [word_embeddings, char_embeddings], dim=-1)
        if self.use_he:
            hnym_embeddings = self.he([hnym, hnym_weights])
            word_embeddings = torch.cat(
                [word_embeddings, hnym_embeddings], dim=-1)
        word_embeddings = self.dropout(word_embeddings)

        if init_hidden is not None:
            hidden = init_hidden
        else:
            hidden = (self.word_linear(word_embeddings), ) * 2

        outputs = []
        for step in range(seq_size):
            inp_seq = seq_embeddings[step, :, :]
            hidden = self.s_lstm(inp_seq, hidden)

            if self.use_i:
                inp_i = torch.cat([word_embeddings, inp_seq], dim=-1)
                hidden = self.s_lstm(inp_i, hidden)

            if self.use_h:
                inp_h = torch.cat([word_embeddings, hidden[0]], dim=-1)
                hidden = (F.tanh(self.h_linear(inp_h)), hidden[1])

            if self.use_g:
                inp_h = torch.cat([word_embeddings, hidden[0]], dim=-1)
                z_t = F.sigmoid(self.g_zt_linear(inp_h))
                r_t = F.sigmoid(self.g_rt_linear(inp_h))
                _hidden = torch.cat([r_t * word_embeddings, hidden[0]], dim=-1)
                _hidden = F.tanh(self.g_ht_linear(_hidden))
                hidden = ((1 - z_t) * hidden[0] + z_t * _hidden, hidden[1])
            outputs.append(hidden[0].view(1, batch_size, -1))
        outputs = torch.cat(outputs, dim=0)
        outputs = self.hidden2tag_linear(outputs)
        outputs = self.dropout(outputs)
        return outputs, hidden


if __name__ == '__main__':
    device = torch.device('cpu')
    word = Variable(torch.randint(0, 999, (100, ), dtype=torch.long))
    seq = Variable(torch.randint(0, 999, (40, 100), dtype=torch.long))
    chars = Variable(torch.randint(0, 999, (100, 15), dtype=torch.long))
    hnym = Variable(torch.randint(0, 999, (100, 5, 2), dtype=torch.long))

    inp = {'word': word, 'seq': seq, 'chars': chars, 'hnym': hnym}

    defseq = DefSeq(
        1000,
        300,
        400,
        device,
        char_vocab_size=1000,
        char_emb_dim=5,
        char_hid_dim=5,
        char_len=15)
    out = defseq(inp)
    print(out.shape)
    out = torch.argmax(out, dim=-1)
    print(out)
    print(out[0].shape)
