# -*- coding: utf-8 -*-
from nltk.translate.bleu_score import corpus_bleu
import json


def read_definition_file(ifp):
    defs = {}
    for line in ifp:
        parts = line.strip().split('\t')
        word = parts[0]
        definition = parts[-1]
        if word not in defs:
            defs[word] = []
        defs[word].append(definition)
    return defs


def make_corpus(refs, hyps):
    refs_corpus = []
    hyps_corpus = []
    for word, definition in hyps.items():
        for d in definition:
            refs_corpus.append([h.split(' ') for h in refs[word]])
            hyps_corpus.append(d.split(' '))
    return refs_corpus, hyps_corpus


def compute_bleu(ref_file, hyp_file):
    print('Reading Files...')
    refs, hyps = None, None
    with open(ref_file) as ifp:
        refs = read_definition_file(ifp)
    with open(hyp_file) as ifp:
        hyps = json.loads(ifp.read())
    refs_corpus, hyps_corpus = make_corpus(refs, hyps)
    print('Computing BLEU...')
    bleu_corpus = corpus_bleu(refs_corpus, hyps_corpus,
                              (0.25, 0.25, 0.25, 0.25))
    print("Corpus Level BLEU: {}".format(bleu_corpus))
    bleu_1 = corpus_bleu(refs_corpus, hyps_corpus, (1, 0, 0, 0))
    print("1-gram BLEU: {}".format(bleu_1))
    bleu_2 = corpus_bleu(refs_corpus, hyps_corpus, (0, 1, 0, 0))
    print("2-gram BLEU: {}".format(bleu_2))
    bleu_3 = corpus_bleu(refs_corpus, hyps_corpus, (0, 0, 1, 0))
    print("3-gram BLEU: {}".format(bleu_3))
    bleu_4 = corpus_bleu(refs_corpus, hyps_corpus, (0, 0, 0, 1))
    print("4-gram BLEU: {}".format(bleu_4))


if __name__ == '__main__':
    ref_file = './data/commondefs/test.txt'
    hyp_file = './output/rerank_output.js'
    compute_bleu(ref_file, hyp_file)
