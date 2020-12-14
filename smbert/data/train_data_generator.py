import random
import pickle

from config import *
from roberta.common.tokenizers import Tokenizer


class TrainDataGenerator(object):
    def __init__(self, oce_corpus_path, ocn_corpus_path, tnews_corpus_path, c2n_pickle_path):
        self.oce_data_tuple = []
        self.ocn_data_tuple = []
        self.tnews_data_tuple = []
        self.tokenizer = Tokenizer(CharsVocabPath)
        with open(c2n_pickle_path, 'rb') as f:
            self.classes2num = pickle.load(f)

        with open(oce_corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    line = line.split('\t')
                    if line[0] and line[1]:
                        self.oce_data_tuple.append([self.classes2num[line[0]], line[1]])
        with open(ocn_corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    line = line.split('\t')
                    if line[0] and line[1]:
                        self.ocn_data_tuple.append([self.classes2num[line[0]]-7, line[1]])
        with open(tnews_corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    line = line.split('\t')
                    if line[0] and line[1]:
                        self.tnews_data_tuple.append([self.classes2num[line[0]]-10, line[1]])

        self.source_oce_data = self.oce_data_tuple
        self.source_ocn_data = self.ocn_data_tuple
        self.source_tnews_data = self.tnews_data_tuple

        random.shuffle(self.oce_data_tuple)
        random.shuffle(self.ocn_data_tuple)
        random.shuffle(self.tnews_data_tuple)

    def get_length(self):
        return len(self.oce_data_tuple), len(self.ocn_data_tuple), len(self.tnews_data_tuple)

    def ret_batch(self):
        self.oce_data_tuple = self.source_oce_data
        self.ocn_data_tuple = self.source_ocn_data
        self.tnews_data_tuple = self.source_tnews_data
        random.shuffle(self.oce_data_tuple)
        random.shuffle(self.ocn_data_tuple)
        random.shuffle(self.tnews_data_tuple)

    def gen_next_batch(self, oce_batch_size, ocn_batch_size, tnews_batch_size):
        output = {}
        batch_max_len = 0
        if len(self.oce_data_tuple) >= oce_batch_size and \
                len(self.ocn_data_tuple) >= ocn_batch_size and \
                len(self.tnews_data_tuple) >= tnews_batch_size:

            oce_current_tuple = self.oce_data_tuple[:oce_batch_size]
            ocn_current_tuple = self.ocn_data_tuple[:ocn_batch_size]
            tnews_current_tuple = self.tnews_data_tuple[:tnews_batch_size]

            self.oce_data_tuple = self.oce_data_tuple[oce_batch_size:]
            self.ocn_data_tuple = self.ocn_data_tuple[ocn_batch_size:]
            self.tnews_data_tuple = self.tnews_data_tuple[tnews_batch_size:]
        else:
            return None

        type_list = []
        label_list = []
        tokens_list = []
        segments_list = []

        for x in oce_current_tuple:
            type_list.append([0])
            label_list.append(x[0])
            token_ids = self.tokenizer.tokens_to_ids(['[CLS]'] + x[1].split(' '))
            if len(token_ids) > batch_max_len:
                batch_max_len = len(token_ids)
            tokens_list.append(token_ids)
            segments_list.append([1] * len(token_ids))
        for x in ocn_current_tuple:
            type_list.append([1])
            label_list.append(x[0])
            token_ids = self.tokenizer.tokens_to_ids(['[CLS]'] + x[1].split(' '))
            if len(token_ids) > batch_max_len:
                batch_max_len = len(token_ids)
            tokens_list.append(token_ids)
            segments_list.append([1] * len(token_ids))
        for x in tnews_current_tuple:
            type_list.append([2])
            label_list.append(x[0])
            token_ids = self.tokenizer.tokens_to_ids(['[CLS]'] + x[1].split(' '))
            if len(token_ids) > batch_max_len:
                batch_max_len = len(token_ids)
            tokens_list.append(token_ids)
            segments_list.append([1] * len(token_ids))

        batch_max_len = min(batch_max_len, SentenceLength)

        for i, tokens in enumerate(tokens_list):
            if len(tokens) < batch_max_len:
                tokens_list[i] = tokens_list[i] + [0] * (batch_max_len - len(tokens))
                segments_list[i] = segments_list[i] + [0] * (batch_max_len - len(tokens))
            else:
                tokens_list[i] = tokens_list[i][:batch_max_len]
                segments_list[i] = segments_list[i][:batch_max_len]

        output['type_id'] = type_list
        output['input_token_ids'] = tokens_list
        output['position_ids'] = [[x for x in range(batch_max_len)] for i in range(oce_batch_size + ocn_batch_size + tnews_batch_size)]
        output['segment_ids'] = segments_list
        output['token_ids_labels'] = label_list
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance


class EvalDataGenerator(object):
    def __init__(self, corpus_path, c2n_pickle_path):
        self.data_tuple = []
        self.corpus_path = corpus_path
        if self.corpus_path == OceEvalPath:
            self.type_id = 0
        if self.corpus_path == OcnEvalPath:
            self.type_id = 1
        if self.corpus_path == TnewsEvalPath:
            self.type_id = 2
        self.tokenizer = Tokenizer(CharsVocabPath)
        with open(c2n_pickle_path, 'rb') as f:
            self.classes2num = pickle.load(f)
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    line = line.split('\t')
                    if line[0] and line[1]:
                        self.data_tuple.append([self.classes2num[line[0]], line[1]])
        self.source_eval_data = self.data_tuple
        random.shuffle(self.data_tuple)

    def reset_batch(self):
        self.data_tuple = self.source_eval_data
        random.shuffle(self.data_tuple)

    def gen_next_batch(self, batch_size):
        output = {}
        batch_max_len = 0
        if len(self.data_tuple) >= batch_size:
            current_tuple = self.data_tuple[:batch_size]
            self.data_tuple = self.data_tuple[batch_size:]
        else:
            return None

        label_list = []
        tokens_list = []
        segments_list = []

        for x in current_tuple:
            if self.type_id == 0:
                label_list.append(x[0])
            if self.type_id == 1:
                label_list.append(x[0] - 7)
            if self.type_id == 2:
                label_list.append(x[0] - 10)
            token_ids = self.tokenizer.tokens_to_ids(['[CLS]'] + x[1].split(' '))
            if len(token_ids) > batch_max_len:
                batch_max_len = len(token_ids)
            tokens_list.append(token_ids)
            segments_list.append([1] * len(token_ids))

        batch_max_len = min(batch_max_len, SentenceLength)

        for i, tokens in enumerate(tokens_list):
            if len(tokens) < batch_max_len:
                tokens_list[i] = tokens_list[i] + [0] * (batch_max_len - len(tokens))
                segments_list[i] = segments_list[i] + [0] * (batch_max_len - len(tokens))
            else:
                tokens_list[i] = tokens_list[i][:batch_max_len]
                segments_list[i] = segments_list[i][:batch_max_len]

        output['type_id'] = [self.type_id]
        output['input_token_ids'] = tokens_list
        output['position_ids'] = [[x for x in range(batch_max_len)]]
        output['segment_ids'] = segments_list
        output['token_ids_labels'] = label_list
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance
