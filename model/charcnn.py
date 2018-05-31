# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-04-26 13:21:40
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    def __init__(self, alphabet_size, pretrain_char_embedding, embedding_dim,
                 hidden_dim, dropout, device):
        super(CharCNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        if pretrain_char_embedding is not None:
            self.char_embeddings.weight.data.copy_(
                torch.from_numpy(pretrain_char_embedding).to(device))
        else:
            self.char_embeddings.weight.data.copy_(
                torch.from_numpy(
                    self.random_embedding(alphabet_size,
                                          embedding_dim)).to(device))
        self.char_cnn = nn.Conv1d(
            embedding_dim, self.hidden_dim, kernel_size=3, padding=1)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale,
                                                       [1, embedding_dim])
        return pretrain_emb

    def get_last_hiddens(self, inp):
        """
            inp:
                inp: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is
            recorded in seq_lengths
        """
        batch_size = inp.size(0)
        char_embeds = self.char_drop(self.char_embeddings(inp))
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds)
        char_cnn_out = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).view(
            batch_size, -1)
        return char_cnn_out

    def get_all_hiddens(self, inp):
        """
            inp:
                inp: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, word_length, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is
            recorded in seq_lengths
        """
        batch_size = inp.size(0)
        char_embeds = self.char_drop(self.char_embeddings(inp))
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds).transpose(2, 1).contiguous()
        return char_cnn_out.view(batch_size, -1)

    def forward(self, inp):
        return self.get_all_hiddens(inp)


if __name__ == "__main__":
    device = torch.device('cpu')
    charcnn = CharCNN(500, None, 5, 5, 0, device)
    inp = torch.autograd.Variable(
        torch.randint(0, 499, (5, 10), dtype=torch.long))
    out = charcnn(inp)
    print(out)
    print(out.shape)
