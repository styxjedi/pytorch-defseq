# -*- coding: utf-8 -*-
import torch
from torch import nn


class Hypernym(nn.Module):
    def __init__(self, emb_dim, emb_layer, device):
        super(Hypernym, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.embedding = emb_layer

    def forward(self, inputs):
        batch_hnym = inputs[0]
        batch_hnym_weights = inputs[1]
        batch_sum = []
        for hnym, weights in zip(batch_hnym, batch_hnym_weights):
            weighted_sum = torch.zeros(300).to(self.device)
            for h, w in zip(hnym, weights):
                word_emb = self.embedding(h)
                weighted_sum += w * word_emb.view(-1)
            batch_sum.append(weighted_sum.expand(1, -1))
        return torch.cat(batch_sum, dim=0)


if __name__ == "__main__":
    device = torch.device('cpu')
    test_inp = [[(23, 149), (34, 135), (42, 132), (52, 82)]] * 5
    # test_inp = [(Variable(torch.LongTensor([i[0]])), Variable(
    #     torch.Tensor([i[1]]))) for i in test_inp]
    test_inp = torch.tensor(test_inp)
    print(test_inp.shape)
    emb_layer = torch.nn.Embedding(100, 300)
    hyper = Hypernym(300, emb_layer, device)
    out = hyper(test_inp)
    print(out.shape)
