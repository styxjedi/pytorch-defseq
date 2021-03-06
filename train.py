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
import sys
# from test import test
# from rerank import rerank
# from nltk_bleu import compute_bleu

BATCH_SIZE = 50
EMB_DIM = 300
HID_DIM = 300
DROPOUT = 0
LEARNING_RATE = 0.001
CHAR_EMB_DIM = 20
CHAR_HID_DIM = 20
CLIP_GRAD = 5


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
    model.training = False
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        losses = []
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
            # target_pred = model(inp)[0]
            loss = loss_fn(target_pred, target)
            losses.append(loss.item())
    return np.mean(losses), np.exp(np.mean(losses))


def main(argv=None):
    if argv is None:
        argv = sys.argv
        if len(argv) == 1:
            model_save_path = 'saved_model/'
        elif len(argv) == 2:
            model_save_path = argv[1]
        else:
            print("Wrong Argument Num. Expected at most 1.")

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        'char_emb_dim': CHAR_EMB_DIM,
        'char_hid_dim': CHAR_HID_DIM,
        'char_len': char_max_len
    }

    model = DefSeq(
        len(word2idx) + 1,
        EMB_DIM,
        HID_DIM,
        device,
        pretrain_emb,
        dropout=DROPOUT,
        **char_data).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    nn.utils.clip_grad_norm()

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    patient = 0
    last_ppl = 0
    min_ppl = 99999
    for epoch in range(100):
        model.training = True
        loss_epoch = []
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
            # target_pred = model(inp)[0]
            loss = loss_fn(target_pred, target)
            loss_epoch.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
        train_loss = np.mean(loss_epoch)
        train_ppl = np.exp(np.mean(loss_epoch))
        valid_loss, valid_ppl = valid(model, valid_loader, device)
        print(
            'Epoch: %03d  train_loss: %.5f  train_ppl: %.5f' % (
                epoch + 1,
                float(train_loss),
                float(train_ppl),
            ),
            end='')
        print('  valid_loss: %.5f  valid_ppl: %.5f' % (float(valid_loss),
                                                       float(valid_ppl)))
        # test('/tmp', 'data/processed/test.npz', device, model=model)
        # rerank('/tmp/output_lines.js', '/tmp/output_scores.js',
        #        '/tmp/rerank_output.js')
        # compute_bleu('data/commondefs/test.txt', '/tmp/rerank_output.js')

        if last_ppl >= valid_ppl:
            patient = 0
        else:
            patient += 1
        last_ppl = valid_ppl

        if (epoch + 1) % 5 == 0:
            torch.save(
                model.state_dict(),
                model_save_path + 'defseq_model_params_%s.pkl' % (epoch + 1))

        if min_ppl > valid_ppl:
            min_ppl = valid_ppl
            torch.save(
                model.state_dict(), model_save_path +
                'defseq_model_params_%s_max_acc.pkl' % (epoch + 1))

        if patient >= 4:
            torch.save(
                model.state_dict(), model_save_path +
                'defseq_model_params_%s_early_stop.pkl' % (epoch + 1))
            break
    return 1


if __name__ == '__main__':
    print('Batch Size:     {}'.format(BATCH_SIZE))
    print('Embedding Dim:  {}'.format(EMB_DIM))
    print('Hidden dim:     {}'.format(HID_DIM))
    print('Dropout Rate:   {}'.format(DROPOUT))
    print('Learning Rate:  {}'.format(LEARNING_RATE))
    print('L2 Regularizer: {}'.format(L2_RATE))
    sys.exit(main())
