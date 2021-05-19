import copy
import random
import numpy as np

from tqdm import tqdm
from random import choice
from smbert.common.tokenizers import Tokenizer
from config import *
from torch.utils.data import Dataset


class DataFactory(object):
    def __init__(self):
        self.tokenizer = Tokenizer(VocabPath)
        self.vocab_size = self.tokenizer._vocab_size
        self.token_pad_id = self.tokenizer._token_pad_id
        self.token_cls_id = self.tokenizer._token_start_id
        self.token_sep_id = self.tokenizer._token_end_id
        self.token_mask_id = self.tokenizer._token_mask_id

    def __token_process(self, token_id):
        """
        以80%的几率替换为[MASK]，以10%的几率保持不变，
        以10%的几率替换为一个随机token。
        """
        rand = np.random.random()
        if rand <= 0.8:
            return self.token_mask_id
        elif rand <= 0.9:
            return token_id
        else:
            return np.random.randint(0, self.vocab_size)

    def ids_to_mask(self, texts_ids):
        instances = []
        tmp_ids = []
        tmp_masks = []
        # 为每个字或者词生成一个概率，用于判断是否mask
        mask_rates = np.random.random(len(texts_ids))

        for i, word_id in enumerate(texts_ids):
            # 为每个字生成对应概率
            tmp_ids.append(word_id)
            if mask_rates[i] < MaskRate:
                tmp_masks.append(self.__token_process(word_id))
            else:
                tmp_masks.append(0)
        tmp_ids = [101] + tmp_ids + [102]
        tmp_masks = [0] + tmp_masks + [0]
        instances.append([tmp_ids, tmp_masks])
        return instances

    def ids_all_mask(self, texts_ids, tokenid2count):
        instances = []
        tmp_ids = [101]

        # 格式化数据
        for token_ids in texts_ids:
            if isinstance(token_ids, list):
                for token_id in token_ids:
                    tmp_ids.append(token_id)
                    if len(tmp_ids) == SentenceLength - 1:
                        break
            else:
                tmp_ids.append(token_ids)
                if len(tmp_ids) == SentenceLength - 1:
                    break
            if len(tmp_ids) == SentenceLength - 1:
                break

        tmp_ids.append(102)
        input_length = len(tmp_ids) - 2

        for i in range(1, input_length + 1):
            # 如果某字出现次数很少，则强行增加训练集
            if tokenid2count[tmp_ids[i]] < WordGenTimes:
                for j in range(WordGenTimes - tokenid2count[tmp_ids[i]]):
                    tmp_masks = [0] * len(tmp_ids)
                    tmp_isG = [0] * len(tmp_ids)
                    # rand_num = np.random.randint(672, 7992)
                    rand_num = choice([m for m in range(672, 7992) if m not in [tmp_ids[i]]])
                    tmp_masks[i] = rand_num
                    tmp_isG[i] = 1
                    instances.append([copy.deepcopy(tmp_ids), copy.deepcopy(tmp_masks), copy.deepcopy(tmp_isG)])
            tmp_masks = [0] * len(tmp_ids)
            tmp_isG = [0] * len(tmp_ids)
            if random.random() < RanWrongDivisor:
                # rand_num = np.random.randint(672, 7992)
                rand_num = choice([m for m in range(672, 7992) if m not in [tmp_ids[i]]])
                tmp_masks[i] = rand_num
                tmp_isG[i] = 1
            else:
                tmp_masks[i] = tmp_ids[i]
            instances.append([copy.deepcopy(tmp_ids), copy.deepcopy(tmp_masks), copy.deepcopy(tmp_isG)])
        return instances


class SMBertDataSet(Dataset):
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.smbert_data = DataFactory()
        self.tar_lines = []
        self.batch_group = []
        self.tokenid_to_count = {}

        self.__load_data()

    def __load_data(self):
        for texts in tqdm(self.__get_texts()):
            texts_ids = self.smbert_data.tokenizer.tokens_to_ids(list(texts))
            # 收集词频
            for tokenids in texts_ids:
                if isinstance(tokenids, list):
                    for tokenid in tokenids:
                        if tokenid in self.tokenid_to_count:
                            self.tokenid_to_count[tokenid] += 1
                        else:
                            self.tokenid_to_count[tokenid] = 1
                else:
                    tokenid = tokenids
                    if tokenid in self.tokenid_to_count:
                        self.tokenid_to_count[tokenid] += 1
                    else:
                        self.tokenid_to_count[tokenid] = 1
            # 基于batchsize组建instance
            if AllMask:
                instances = self.smbert_data.ids_all_mask(texts_ids, self.tokenid_to_count)
            else:
                instances = self.smbert_data.ids_to_mask(texts_ids)
            for instance in instances:
                self.batch_group.append(instance)
                if len(self.batch_group) == BatchSize:
                    self.tar_lines.append(self.batch_group)
                    self.batch_group = []
        if len(self.batch_group) > 0:
            self.tar_lines.append(self.batch_group)
            self.batch_group = []

    def __get_texts(self):
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                yield line

    def __gen_input_token(self, token_ids, mask_ids):
        assert len(token_ids) == len(mask_ids)
        input_token_ids = []
        for token, mask in zip(token_ids, mask_ids):
            tmp_ids = []
            for i in range(len(token)):
                if mask[i] == 0:
                    tmp_ids.append(token[i])
                else:
                    tmp_ids.append(mask[i])
            input_token_ids.append(tmp_ids)
        return input_token_ids

    def __len__(self):
        return len(self.tar_lines)

    def __getitem__(self, item):
        output = {}
        instances = self.tar_lines[item]
        token_ids = copy.deepcopy([x[0] for x in instances])
        mask_ids = copy.deepcopy([x[1] for x in instances])
        isG_ids = copy.deepcopy([x[2] for x in instances])
        input_token_ids = self.__gen_input_token(token_ids, mask_ids)

        # 构建batch数据
        batch_max = max([len(x) for x in input_token_ids])
        for i in range(len(input_token_ids)):
            for j in range(batch_max - len(input_token_ids[i])):
                input_token_ids[i].append(0)
                token_ids[i].append(0)
                isG_ids[i].append(0)

        # 构建segment_ids
        segment_ids = []
        for x in input_token_ids:
            segment_ids.append([1 if y else 0 for y in x])

        # 构建position_ids
        position_ids = []
        for x in token_ids:
            position_ids.append([y for y in range(batch_max)])

        output['batch_inputs'] = input_token_ids
        output['batch_position'] = position_ids
        output['batch_segments'] = segment_ids
        output['batch_labels'] = token_ids
        output['batch_isG'] = isG_ids
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        instance['batch_isG'] = instance['batch_isG'].float()

        return instance


class SMBertEvalSet(Dataset):
    def __init__(self, eval_path):
        self.tokenizer = Tokenizer(VocabPath)
        self.eval_path = eval_path
        self.eval_lines = []
        self.label_lines = []

        self.__load_data()

    def __load_data(self):
        # 读取数据
        with open(self.eval_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    line_list = line.split('-***-')
                    self.eval_lines.append(line_list[1])
                    self.label_lines.append(line_list[0])

    def __gen_token(self, tokens):
        tar_token_ids = [101]
        tokens = list(tokens)
        tokens = tokens[:(SentenceLength - 2)]
        for token in tokens:
            token_id = self.tokenizer.token_to_id(token)
            tar_token_ids.append(token_id)
        tar_token_ids.append(102)
        return tar_token_ids

    def __len__(self):
        return len(self.label_lines)

    def __getitem__(self, item):
        output = {}
        eval_text = self.eval_lines[item]
        label_text = self.label_lines[item]
        eval_token = self.__gen_token(eval_text)
        label_token = self.__gen_token(label_text)
        position_ids = [i for i in range(len(eval_token))]
        segment_ids = [1 if x else 0 for x in label_token]
        output['eval_token'] = eval_token
        output['eval_position'] = position_ids
        output['eval_segment'] = segment_ids
        output['eval_label'] = label_token
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance


if __name__ == '__main__':
    tt = SMBertDataSet('../../data/train_data/train.txt')
    for e in range(16):
        for i, x in enumerate(tt):
            print(i)
