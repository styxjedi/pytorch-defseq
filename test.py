# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
from model.defseq import DefSeq
from utils.dataset import DefSeqDataset
from utils.beamsearch import BeamSearch
from tqdm import tqdm
import os

EMB_DIM = 300
HID_DIM = 300


def get_test_loader(file_path):
    test = np.load(file_path)
    test_dataset = DefSeqDataset(test, 'test')
    char_max_len = len(test_dataset[0]['chars'])
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4)
    return test_loader, char_max_len


def main():
    output_save_path = './output'
    model_path = './saved_model/defseq_model_params_10.pkl'
    if not os.path.exists(output_save_path):
        os.makedirs(output_save_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using Device: ', device)

    char2idx = json.loads(open('data/processed/char2idx.js').read())
    word2idx = json.loads(open('data/processed/word2idx.js').read())
    idx2word = {v: k for k, v in word2idx.items()}
    pretrain_emb = torch.tensor(
        np.load('data/processed/preptrain_emb.npy')).to(device)
    test_file_path = 'data/processed/test.npz'
    test_loader, char_max_len = get_test_loader(test_file_path)

    char_data = {
        'char_vocab_size': len(char2idx) + 1,
        'char_emb_dim': 5,
        'char_hid_dim': 10,
        'char_len': char_max_len
    }

    model = DefSeq(
        len(word2idx) + 1, EMB_DIM, HID_DIM, device, pretrain_emb,
        **char_data).to(device)
    model.load_state_dict(torch.load(model_path, str(device)))
    beam = BeamSearch(
        400,
        word2idx['<unk>'],
        word2idx['<s>'],
        word2idx['</s>'],
        beam_size=50)
    output_lines = {}
    output_scores = {}
    for feed_dict in tqdm(test_loader, desc='Test', leave=False):
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
        beam.reset()
        probs, hidden = model(inp)
        probs = probs.detach().cpu().numpy().squeeze(0)
        while beam.beam(probs):
            inp['seq'] = torch.tensor(
                beam.live_samples, dtype=torch.long).to(device)
            inp['seq'] = inp['seq'][:, -1].expand(1, -1)
            inp['word'] = inp['word'][0].repeat(inp['seq'].shape[1])
            inp['chars'] = inp['chars'][0].expand(1, -1).repeat(
                inp['seq'].shape[1], 1)
            inp['hnym'] = inp['hnym'][0].expand(1, -1).repeat(
                inp['seq'].shape[1], 1)
            inp['hnym_weights'] = inp['hnym_weights'][0].expand(1, -1).repeat(
                inp['seq'].shape[1], 1)
            hidden = (hidden[0][0].expand(1, -1).repeat(
                inp['seq'].shape[1], 1), hidden[1][0].expand(1, -1).repeat(
                    inp['seq'].shape[1], 1))
            probs, hidden = model(inp, hidden)
            probs = probs.detach().cpu().numpy().squeeze(0)
        line = [[idx2word[i] for i in line if i not in [1, 2, 3]]
                for line in beam.output]
        line = [' '.join(line) for line in line]
        word = idx2word[int(feed_dict['word'])]
        print(word)
        print('\n'.join(line[:5]))
        output_lines[word] = line
        output_scores[word] = beam.output_scores
    with open(output_save_path + '/output_lines.js', 'w') as fw_lines:
        fw_lines.write(json.dumps(output_lines))
    with open(output_save_path + '/output_scores.js', 'w') as fw_scores:
        fw_scores.write(json.dumps(output_scores))
    return 1


if __name__ == '__main__':
    if main():
        print('All Done. No Error.')
