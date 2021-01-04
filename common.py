from config import *


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
