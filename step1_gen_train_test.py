import random

from common import check_srcdata_and_vocab
from config import *
from smbert.common.tokenizers import Tokenizer


def random_wrong(text):
    tokenizer = Tokenizer(VocabPath)
    length = len(text)
    position = random.randint(0, length-1)
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
