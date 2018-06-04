# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import numpy as np
import json
from model.defseq import DefSeq
from utils.dataset import DefSeqDataset
from tqdm import tqdm
import os

BATCH_SIZE = 50
EMB_DIM = 300
HID_DIM = 300


def get_train_loader(file_path):
    train = np.load(file_path)
    train_dataset = DefSeqDataset(train, 'train')
    char_max_len = len(train_dataset[0]['chars'])
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    return train_loader, char_max_len


def get_acc(y_pred, y):
    y_pred = torch.argmax(y_pred, dim=1)
    assert y_pred.shape == y.shape
    total = y.view(-1).shape[0]
    acc_num = torch.sum(y_pred == y)
    acc = float(acc_num) / float(total)
    return acc


def valid(model, valid_loader, device):
    loss_fn = nn.NLLLoss()
    with torch.no_grad():
        loss_all = torch.tensor([0.]).to(device)
        for feed_dict in tqdm(valid_loader, desc='Validation', leave=False):
            inp = {
                'word':
                torch.tensor(feed_dict['word'], dtype=torch.long).to(device),
                'seq':
                torch.tensor(torch.t(feed_dict['seq']),
                             dtype=torch.long).to(device),
                'chars':
                torch.tensor(feed_dict['chars'], dtype=torch.long).to(device),
                'hnym':
                torch.tensor(feed_dict['hnym'], dtype=torch.long).to(device),
                'hnym_weights':
                torch.tensor(feed_dict['hnym_weights'],
                             dtype=torch.float).to(device)
            }
            target = torch.tensor(
                feed_dict['target'], dtype=torch.long).to(device)
            target_pred = model(inp)[0].transpose(0, 1).transpose(1, 2)
            loss = loss_fn(target_pred, target)
            loss_all += loss
            acc = get_acc(target_pred, target)
    return loss_all, acc


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using Device: ', device)

    char2idx = json.loads(open('data/processed/char2idx.js').read())
    word2idx = json.loads(open('data/processed/word2idx.js').read())
    pretrain_emb = torch.tensor(
        np.load('data/processed/preptrain_emb.npy')).to(device)
    train_file_path = 'data/processed/train.npz'
    valid_file_path = 'data/processed/valid.npz'
    train_loader, char_max_len = get_train_loader(train_file_path)
    valid_loader, _ = get_train_loader(valid_file_path)

    char_data = {
        'char_vocab_size': len(char2idx) + 1,
        'char_emb_dim': 5,
        'char_hid_dim': 10,
        'char_len': char_max_len
    }

    model = DefSeq(
        len(word2idx) + 1, EMB_DIM, HID_DIM, device, pretrain_emb,
        **char_data).to(device)
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    model_save_path = 'saved_model/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    patient = 0
    last_acc = 0
    for epoch in range(200):
        loss_epoch = []
        acc_epoch = []
        for feed_dict in tqdm(
                train_loader, desc='Epoch: %03d' % (epoch + 1), leave=False):
            inp = {
                'word':
                torch.tensor(feed_dict['word'], dtype=torch.long).to(device),
                'seq':
                torch.tensor(torch.t(feed_dict['seq']),
                             dtype=torch.long).to(device),
                'chars':
                torch.tensor(feed_dict['chars'], dtype=torch.long).to(device),
                'hnym':
                torch.tensor(feed_dict['hnym'], dtype=torch.long).to(device),
                'hnym_weights':
                torch.tensor(feed_dict['hnym_weights'],
                             dtype=torch.float).to(device)
            }
            target = torch.tensor(
                feed_dict['target'], dtype=torch.long).to(device)
            optimizer.zero_grad()
            target_pred = model(inp)[0].transpose(0, 1).transpose(1, 2)
            # print(target_pred.shape)
            # pdb.set_trace()
            loss = loss_fn(target_pred, target)
            loss_epoch.append(float(loss))
            loss.backward()
            optimizer.step()
            acc_epoch.append(get_acc(target_pred, target))
        train_loss = sum(loss_epoch)
        train_acc = sum(acc_epoch) / len(acc_epoch)
        valid_loss, valid_acc = valid(model, valid_loader, device)
        print(
            'Epoch: %03d  loss: %.5f  acc: %.5f' % (
                epoch + 1,
                float(train_loss),
                float(train_acc),
            ),
            end='')
        print('  valid_loss: %.5f  valid_acc: %.5f' % (float(valid_loss),
                                                       float(valid_acc)))
        if last_acc - valid_acc <= 0:
            patient = 0
        else:
            patient += 1
        if (epoch + 1) % 5 == 0:
            torch.save(
                model.state_dict(),
                model_save_path + 'defseq_model_params_%s.pkl' % (epoch + 1))
        if patient == 5:
            torch.save(
                model.state_dict(),
                model_save_path + 'defseq_model_params_%s.pkl' % (epoch + 1))
            break
    return 1


if __name__ == '__main__':
    if main():
        print('All Done. No Error.')
