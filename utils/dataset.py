# -*- coding: utf-8 -*-
from torch.utils.data import Dataset


class DefSeqDataset(Dataset):
    def __init__(self, data_file, mode='train'):
        self.mode = mode
        if self.mode not in ['train', 'test', 'valid']:
            raise Exception('Argument mode must be train, test or valid.')
        self.word = data_file['word']
        self.seq = data_file['seq']
        self.chars = data_file['chars']
        self.hnym = data_file['hnym']
        self.hnym_weights = data_file['hnym_weights']
        if not self.mode == 'test':
            self.target = data_file['target']

    def __len__(self):
        return len(self.word)

    def __getitem__(self, idx):
        sample = {
            'word': self.word[idx],
            'seq': self.seq[idx],
            'chars': self.chars[idx],
            'hnym': self.hnym[idx],
            'hnym_weights': self.hnym_weights[idx],
        }
        if not self.mode == 'test':
            sample['target'] = self.target[idx]
        return sample
