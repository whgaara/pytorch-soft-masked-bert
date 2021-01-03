import math
import random
import pkuseg
import numpy as np

from tqdm import tqdm
from smbert.common.tokenizers import Tokenizer
from pretrain_config import *
from torch.utils.data import Dataset


class DataFactory(object):
    def __init__(self):
        self.tokenizer = Tokenizer(VocabPath)
        self.seg = pkuseg.pkuseg()
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

    def texts_to_ids(self, texts):
        texts_ids = []
        for text in texts:
            # 处理每个句子
            for word in text:
                # text_ids首位分别是cls和sep，这里暂时去除
                word_tokes = self.tokenizer.tokenize(text=word)[1:-1]
                words_ids = self.tokenizer.tokens_to_ids(word_tokes)
                texts_ids.append(words_ids)
        return texts_ids

    def ids_to_mask(self, texts_ids):
        instances = []
        total_ids = []
        total_masks = []
        # 为每个字或者词生成一个概率，用于判断是否mask
        mask_rates = np.random.random(len(texts_ids))

        for i, word_id in enumerate(texts_ids):
            # 为每个字生成对应概率
            total_ids.extend(word_id)
            if mask_rates[i] < MaskRate:
                # 因为word_id可能是一个字，也可能是一个词
                for sub_id in word_id:
                    total_masks.append(self.__token_process(sub_id))
            else:
                total_masks.extend([0]*len(word_id))

        # 每个实例的最大长度为512，因此对一个段落进行裁剪
        # 510 = 512 - 2，给cls和sep留的位置
        for i in range(math.ceil(len(total_ids)/(SentenceLength - 2))):
            tmp_ids = [self.token_cls_id]
            tmp_masks = [self.token_pad_id]
            tmp_ids.extend(total_ids[i*(SentenceLength - 2): min((i+1)*(SentenceLength - 2), len(total_ids))])
            tmp_masks.extend(total_masks[i*(SentenceLength - 2): min((i+1)*(SentenceLength - 2), len(total_masks))])
            # 不足512的使用padding补全
            diff = SentenceLength - len(tmp_ids)
            if diff == 1:
                tmp_ids.append(self.token_sep_id)
                tmp_masks.append(self.token_pad_id)
            else:
                # 添加结束符
                tmp_ids.append(self.token_sep_id)
                tmp_masks.append(self.token_pad_id)
                # 将剩余部分padding补全
                tmp_ids.extend([self.token_pad_id] * (diff - 1))
                tmp_masks.extend([self.token_pad_id] * (diff - 1))
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
        if len(tmp_ids) < SentenceLength:
            for i in range(SentenceLength - len(tmp_ids)):
                tmp_ids.append(0)

        for i in range(1, input_length + 1):
            # 如果某字出现次数很少，则强行增加训练集
            if tokenid2count[tmp_ids[i]] < WordGenTimes:
                for j in range(WordGenTimes - tokenid2count[tmp_ids[i]]):
                    tmp_masks = [0] * SentenceLength
                    rand_num = np.random.randint(672, 7992)
                    tmp_masks[i] = rand_num
                    instances.append([tmp_ids, tmp_masks])
            tmp_masks = [0] * SentenceLength
            if random.random() < RanWrongDivisor:
                rand_num = np.random.randint(672, 7992)
                tmp_masks[i] = rand_num
            else:
                tmp_masks[i] = tmp_ids[i]
            instances.append([tmp_ids, tmp_masks])
        return instances


class SMBertDataSet(Dataset):
    def __init__(self, corpus_path, onehot_type=False):
        self.corpus_path = corpus_path
        self.onehot_type = onehot_type
        self.smbert_data = DataFactory()
        self.src_lines = []
        self.tar_lines = []
        self.tokenid_to_count = {}
        for i in range(RepeatNum):
            for texts in tqdm(self.__get_texts()):
                texts_ids = self.smbert_data.texts_to_ids(texts)
                self.src_lines.append(texts_ids)
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
        for line in self.src_lines:
            instances = self.smbert_data.ids_all_mask(line, self.tokenid_to_count)
            for instance in instances:
                self.tar_lines.append(instance)

    def __get_texts(self):
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                yield [line]

    def __len__(self):
        return len(self.tar_lines)

    def __getitem__(self, item):
        output = {}
        instance = self.tar_lines[item]
        token_ids = instance[0]
        mask_ids = instance[1]
        is_masked = []
        for i, id in enumerate(mask_ids):
            if id != 0:
                is_masked.append(i)
        input_token_ids = self.__gen_input_token(token_ids, mask_ids)
        segment_ids = [1 if x else 0 for x in token_ids]
        if self.onehot_type:
            # 全体onehot生成非常耗时，暂时注释，需要时可使用
            onehot_labels = self.__id_to_onehot(token_ids)
            # 只针对mask结果进行onehot
            mask_onehot_labels = self.__maskid_to_onehot(token_ids, is_masked)
            output['onehot_labels'] = onehot_labels
            output['mask_onehot_labels'] = mask_onehot_labels

        output['input_token_ids'] = input_token_ids
        output['token_ids_labels'] = token_ids
        output['is_masked'] = is_masked
        output['segment_ids'] = segment_ids

        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance

    def __gen_input_token(self, token_ids, mask_ids):
        assert len(token_ids) == len(mask_ids)
        input_token_ids = []
        for token, mask in zip(token_ids, mask_ids):
            if mask == 0:
                input_token_ids.append(token)
            else:
                input_token_ids.append(mask)
        return input_token_ids

    def __id_to_onehot(self, token_ids):
        onehot_labels = []
        onehot_pad = [0] * VocabSize
        onehot_pad[0] = 1
        for i in token_ids:
            tmp = [0 for j in range(VocabSize)]
            if i == 0:
                onehot_labels.append(onehot_pad)
            else:
                tmp[i] = 1
                onehot_labels.append(tmp)
        return onehot_labels

    def __maskid_to_onehot(self, token_ids, is_masked):
        onehot_masked_labels = []
        for i in is_masked:
            onehot_labels = [0] * VocabSize
            onehot_labels[token_ids[i]] = 1
            onehot_masked_labels.append(onehot_labels)
        return onehot_masked_labels


class SMBertEvalSet(Dataset):
    def __init__(self, test_path):
        self.tokenizer = Tokenizer(VocabPath)
        self.test_path = test_path
        self.test_lines = []
        self.label_lines = []
        # 读取数据
        with open(self.test_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    line_list = line.split('-***-')
                    self.test_lines.append(line_list[1])
                    self.label_lines.append(line_list[0])

    def __len__(self):
        return len(self.label_lines)

    def __getitem__(self, item):
        output = {}
        test_text = self.test_lines[item]
        label_text = self.label_lines[item]
        test_token = self.__gen_token(test_text)
        label_token = self.__gen_token(label_text)
        segment_ids = [1 if x else 0 for x in label_token]
        output['input_token_ids'] = test_token
        output['token_ids_labels'] = label_token
        output['segment_ids'] = segment_ids
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance

    def __gen_token(self, tokens):
        tar_token_ids = [101]
        tokens = list(tokens)
        tokens = tokens[:(SentenceLength - 2)]
        for token in tokens:
            token_id = self.tokenizer.token_to_id(token)
            tar_token_ids.append(token_id)
        tar_token_ids.append(102)
        if len(tar_token_ids) < SentenceLength:
            for i in range(SentenceLength - len(tar_token_ids)):
                tar_token_ids.append(0)
        return tar_token_ids
