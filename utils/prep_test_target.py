# -*- coding: utf-8 -*-
from collections import defaultdict
import json


def read_origin_data(file_path):
    content = []
    with open(file_path, 'r') as fr:
        for line in fr:
            line = line.strip().split('\t')
            content.append((line[0], line[-1]))
    return content


def main():
    test_read_path = '../data/commondefs/test.txt'
    test_data = read_origin_data(test_read_path)
    test_target = defaultdict(list)
    for word, seq in test_data:
        test_target[word].append(seq)
    with open('../data/processed/test_target.js', 'w') as fw:
        fw.write(json.dumps(test_target))
    return 1


if __name__ == "__main__":
    if main():
        print("All Done. No Error.")
