import pkuseg
from pretrain_config import *


def check_srcdata_and_vocab(target_path):
    segment = pkuseg.pkuseg()
    f1 = open(target_path, 'r', encoding='utf-8')
    f2 = open(VocabPath, 'r', encoding='utf-8')
    local_tokens = []
    vocabs = []
    missing = []
    if ModelClass == 'SMBertMlm':
        for l in f1:
            if l:
                l = l.strip()
                l_seg = segment.cut(l)
                for x in l_seg:
                    local_tokens.append(x)
    else:
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
