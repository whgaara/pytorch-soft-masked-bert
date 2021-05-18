import random

from config import *
from smbert.common.tokenizers import Tokenizer


def check_srcdata_and_vocab(target_path):
    f1 = open(target_path, 'r', encoding='utf-8')
    f2 = open(VocabPath, 'r', encoding='utf-8')
    local_tokens = []
    vocabs = []
    missing = []
    for l in f1:
        if l:
            l = l.strip()
            for x in l:
                local_tokens.append(x)
    local_tokens = list(set(local_tokens))

    for l in f2:
        if l:
            l = l.strip()
            vocabs.append(l)
    for x in local_tokens:
        if x not in vocabs:
            missing.append(x)
    if missing:
        print('警告！本地vocab缺少以下字符：')
        for x in missing:
            print(x)


def random_wrong(text):
    tokenizer = Tokenizer(VocabPath)
    length = len(text)
    position = random.randint(0, length - 1)
    number = random.randint(672, 7992)
    text = list(text)
    text[position] = tokenizer.id_to_token(number)
    text = ''.join(text)
    return text


def gen_train_test():
    f_train = open(CorpusPath, 'w', encoding='utf-8')
    f_test = open(TestPath, 'w', encoding='utf-8')
    with open(SourcePath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            rad = random.randint(0, 10)
            if rad < 1:
                f_test.write(line + '-***-' + random_wrong(line) + '\n')
                f_train.write(line + '\n')
            else:
                f_train.write(line + '\n')

    f_train.close()
    f_test.close()


if __name__ == '__main__':
    print(len(open(VocabPath, 'r', encoding='utf-8').readlines()))
    check_srcdata_and_vocab(SourcePath)
    gen_train_test()
