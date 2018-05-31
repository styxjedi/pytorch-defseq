# -*- coding: utf-8 -*-
from collections import defaultdict
import os
import gensim
import numpy as np
import json
from tqdm import tqdm


def read_origin_data(file_path):
    content = []
    with open(file_path, 'r') as fr:
        for line in fr:
            line = line.strip().split('\t')
            content.append((line[0], line[-1]))
    return content


def read_hypernyms(file_path):
    hnym_data = defaultdict(list)
    with open(file_path, 'r') as fr:
        for line in fr:
            line = line.strip().split('\t')
            word = line[0]
            line = line[1:]
            assert len(line) % 2 == 0
            for i in range(int(len(line) / 2)):
                hnym = line[2 * i]
                weight = line[2 * i + 1]
                hnym_data[word].append((hnym, weight))
    return hnym_data


def make_vocab(train, test, valid, hnym):
    vocab = set()
    for data in [train, test, valid]:
        for word, seq in data:
            if word not in vocab:
                vocab.add(word)
            for w in seq.split(' '):
                if w not in vocab:
                    vocab.add(w)
    for k, v in hnym.items():
        if k not in vocab:
            vocab.add(k)
        for hnym, weight in v:
            if hnym not in vocab:
                vocab.add(hnym)
    return vocab


def make_word2idx(vocab):
    word2idx = {'<s>': 1, '</s>': 2, '<unk>': 3}
    for w in vocab:
        if w not in word2idx:
            word2idx[w] = len(word2idx) + 1

    char2idx = {'<c>': 1, '</c>': 2}
    for w in vocab:
        for c in w:
            if c not in char2idx:
                char2idx[c] = len(char2idx) + 1
    return word2idx, char2idx


def get_pretrain_emb(word2vec_path, word2idx):
    print('Making Pretrained Matrix...')
    model = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_path, binary=True)
    pretrain_emb = np.random.random((len(word2idx) + 1, model.vector_size))
    pretrain_emb[0] = np.zeros((300, ))
    for word in tqdm(word2idx.keys(), leave=False):
        try:
            pretrain_emb[word2idx[word]] = model.word_vec(word)
        except KeyError:
            pass
    del model
    print('Done!')
    return pretrain_emb


def get_hnym(hnym_data, word2idx, top_k=5):
    word2hnym = defaultdict(list)
    hnym_weights = defaultdict(list)
    for key, value in hnym_data.items():
        weight_sum = sum([float(w) for h, w in value])
        for hnym, weight in value:
            word2hnym[key].append(word2idx[hnym])
            hnym_weights[key].append(float(weight) / weight_sum)
    return word2hnym, hnym_weights


def prep_word(train, test, valid, word2idx):
    print('Preparing Words...')
    train_word = np.zeros((len(train), ))
    test_word = np.zeros((len(test), ))
    valid_word = np.zeros((len(valid), ))

    for data, norm_word in zip([train, test, valid],
                               [train_word, test_word, valid_word]):
        assert len(data) == len(norm_word)
        for i, (word, _) in tqdm(enumerate(data), leave=False):
            norm_word[i] = word2idx[word]
    print('Done!')
    return train_word, test_word, valid_word


def prep_seq(train, test, valid, word2idx):
    print('Preparing Sequences...')
    max_len = 0
    for data in [train, valid]:
        for _, seq in data:
            seq_list = seq.split(' ')
            max_len = max(max_len, len(seq_list) + 2)
    train_seq = np.zeros((len(train), max_len))
    valid_seq = np.zeros((len(valid), max_len))
    for data, norm_seq in zip([train, valid], [train_seq, valid_seq]):
        assert len(data) == len(norm_seq)
        for i, (_, seq) in tqdm(enumerate(data), leave=False):
            seq = seq.split(' ')
            seq.insert(0, '<s>')
            seq.append('</s>')
            for j, w in enumerate(seq):
                norm_seq[i][j] = word2idx[w]

    test_seq = np.zeros((len(test), 1))
    for i in range(len(test)):
        test_seq[i] = np.array([word2idx['<s>']])
    print('Done!')
    return train_seq, test_seq, valid_seq


def prep_chars(train, test, valid, char2idx):
    print('Preparing Characters...')
    max_len = 0
    for data in [train, test, valid]:
        for word, _ in data:
            word_len = len(word) + 2
            max_len = max(word_len, max_len)
    train_chars = np.zeros((len(train), max_len))
    test_chars = np.zeros((len(test), max_len))
    valid_chars = np.zeros((len(valid), max_len))
    for data, norm_chars in zip([train, test, valid],
                                [train_chars, test_chars, valid_chars]):
        assert len(data) == len(norm_chars)
        for i, (word, _) in tqdm(enumerate(data), leave=False):
            chars = [c for c in word]
            chars.insert(0, '<c>')
            chars.append('</c>')
            for j, c in enumerate(chars):
                norm_chars[i][j] = char2idx[c]
    print('Done!')
    return train_chars, test_chars, valid_chars


def prep_hnym(train, test, valid, word2hnym, hnym_weights, top_k=5):
    print('Preparing Hypernyms...')
    train_hnym = np.zeros((len(train), top_k))
    test_hnym = np.zeros((len(test), top_k))
    valid_hnym = np.zeros((len(valid), top_k))
    train_hnym_weights = np.zeros_like(train_hnym)
    test_hnym_weights = np.zeros_like(test_hnym)
    valid_hnym_weights = np.zeros_like(valid_hnym)
    for data, norm_hnym, norm_hnym_weights in zip(
        [train, test, valid], [train_hnym, test_hnym, valid_hnym],
        [train_hnym_weights, test_hnym_weights, valid_hnym_weights]):
        assert len(data) == len(norm_hnym)
        for i, (word, _) in tqdm(enumerate(data), leave=False):
            for j, hnym in enumerate(word2hnym[word][:top_k]):
                norm_hnym[i][j] = hnym
            for k, weight in enumerate(hnym_weights[word][:top_k]):
                norm_hnym_weights[i][k] = weight
    print('Done!')
    return train_hnym, train_hnym_weights, test_hnym, test_hnym_weights, valid_hnym, valid_hnym_weights


def prep_target(train, test, valid, word2idx):
    print('Preparing Targets...')
    max_len = 0
    for data in [train, valid]:
        for _, seq in data:
            seq_list = seq.split(' ')
            max_len = max(max_len, len(seq_list) + 2)
    train_target = np.zeros((len(train), max_len))
    valid_target = np.zeros((len(valid), max_len))
    for data, norm_target in zip([train, valid], [train_target, valid_target]):
        assert len(data) == len(norm_target)
        for i, (_, seq) in tqdm(enumerate(data), leave=False):
            seq = seq.split(' ')
            seq.append('</s>')
            for j, w in enumerate(seq):
                norm_target[i][j] = word2idx[w]
    test_target = defaultdict(list)
    for word, seq in test:
        test_target[word].append(seq)
    print('Done!')
    return train_target, test_target, valid_target


def main():
    print('Reading Origin Data...')
    train_data = read_origin_data('../data/commondefs/train.txt')
    test_data = read_origin_data('../data/commondefs/test.txt')
    valid_data = read_origin_data('../data/commondefs/valid.txt')
    hnym_data = read_hypernyms(
        '../data/commondefs/auxiliary/bag_of_hypernyms.txt')
    vocab = make_vocab(train_data, test_data, valid_data, hnym_data)
    print('Done!')
    word2idx, char2idx = make_word2idx(vocab)
    pretrain_emb = get_pretrain_emb(
        '../data/GoogleNews-vectors-negative300.bin', word2idx)
    top_k = 5
    word2hnym, hnym_weights = get_hnym(hnym_data, word2idx, top_k)

    train_word, test_word, valid_word = prep_word(train_data, test_data,
                                                  valid_data, word2idx)
    train_seq, test_seq, valid_seq = prep_seq(train_data, test_data,
                                              valid_data, word2idx)
    train_chars, test_chars, valid_chars = prep_chars(train_data, test_data,
                                                      valid_data, char2idx)
    train_hnym, train_hnym_weights, test_hnym, test_hnym_weights, valid_hnym, valid_hnym_weights = prep_hnym(
        train_data, test_data, valid_data, word2hnym, hnym_weights, top_k)
    train_target, test_target, valid_target = prep_target(
        train_data, test_data, valid_data, word2idx)

    print('Saving to Files...')
    prep_path = '../data/processed/'
    if not os.path.exists(prep_path):
        os.makedirs(prep_path)

    np.save(prep_path + 'preptrain_emb', pretrain_emb)
    np.savez_compressed(
        prep_path + 'train',
        word=train_word,
        seq=train_seq,
        chars=train_chars,
        hnym=train_hnym,
        hnym_weights=train_hnym_weights,
        target=train_target)
    np.savez_compressed(
        prep_path + 'valid',
        word=valid_word,
        seq=valid_seq,
        chars=valid_chars,
        hnym=valid_hnym,
        hnym_weights=valid_hnym_weights,
        target=valid_target)
    np.savez_compressed(
        prep_path + 'test',
        word=test_word,
        seq=test_seq,
        chars=test_chars,
        hnym=test_hnym,
        hnym_weights=test_hnym_weights)

    with open(prep_path + 'word2idx.js', 'w') as fr1:
        fr1.write(json.dumps(word2idx))
    with open(prep_path + 'word2hnym.js', 'w') as fr2:
        fr2.write(json.dumps(word2hnym))
    with open(prep_path + 'char2idx.js', 'w') as fr3:
        fr3.write(json.dumps(char2idx))
    with open(prep_path + 'test_target.js', 'w') as fr4:
        fr4.write(json.dumps(test_target))
    with open(prep_path + 'hnym_weights.js', 'w') as fr5:
        fr5.write(json.dumps(hnym_weights))
    print('Done!')
    return 1


if __name__ == '__main__':
    if main():
        print('All Done. No Error.')
