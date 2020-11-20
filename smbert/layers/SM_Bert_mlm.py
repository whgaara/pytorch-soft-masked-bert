import os
import torch.nn as nn

from pretrain_config import *
from smbert.common.tokenizers import Tokenizer
from smbert.layers.Transformer import Transformer
from smbert.layers.SMBertEmbeddings import SMBbertEmbeddings
from smbert.layers.Mlm import Mlm
from smbert.layers.BiGRU import BiGRU


class SMBertMlm(nn.Module):
    def __init__(self,
                 vocab_size=VocabSize,
                 hidden=HiddenSize,
                 max_len=SentenceLength,
                 num_hidden_layers=HiddenLayerNum,
                 attention_heads=AttentionHeadNum,
                 dropout_prob=DropOut,
                 intermediate_size=IntermediateSize
                 ):
        super(SMBertMlm, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden
        self.max_len = max_len
        self.num_hidden_layers = num_hidden_layers
        self.attention_head_num = attention_heads
        self.dropout_prob = dropout_prob
        self.attention_head_size = hidden // attention_heads
        self.tokenizer = Tokenizer(VocabPath)
        self.intermediate_size = intermediate_size

        # 申明网络
        self.smbert_emd = SMBbertEmbeddings(vocab_size=self.vocab_size, max_len=self.max_len, hidden_size=self.hidden_size)
        self.bi_gru = BiGRU(self.hidden_size, self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.transformer_blocks = nn.ModuleList(
            Transformer(
                hidden_size=self.hidden_size,
                attention_head_num=self.attention_head_num,
                attention_head_size=self.attention_head_size,
                intermediate_size=self.intermediate_size).to(device)
            for _ in range(self.num_hidden_layers)
        )
        self.mlm = Mlm(self.hidden_size, self.vocab_size)

    @staticmethod
    def gen_attention_masks(segment_ids):
        def gen_attention_mask(segment_id):
            dim = segment_id.size()[-1]
            attention_mask = torch.zeros([dim, dim], dtype=torch.int64)
            end_point = 0
            for i, segment in enumerate(segment_id.tolist()):
                if segment:
                    end_point = i
                else:
                    break
            for i in range(end_point + 1):
                for j in range(end_point + 1):
                    attention_mask[i][j] = 1
            return attention_mask
        attention_masks = []
        segment_ids = segment_ids.tolist()
        for segment_id in segment_ids:
            attention_mask = gen_attention_mask(torch.tensor(segment_id))
            attention_masks.append(attention_mask.tolist())
        return torch.tensor(attention_masks)

    def load_pretrain(self, path=PretrainPath):
        pretrain_model_dict = torch.load(path)
        finetune_model_dict = pretrain_model_dict.state_dict()
        self.load_state_dict(finetune_model_dict)

    def forward(self, input_token, segment_ids):
        # embedding
        if Debug:
            print('获取embedding %s' % get_time())
        embedding_x, mask_embedding_x = self.smbert_emd(input_token, segment_ids)
        if Debug:
            print('获取attention_mask %s' % get_time())

        # error detection
        bi_gru_x = self.bi_gru(embedding_x)
        pi = self.sigmoid(bi_gru_x)
        embedding_i = pi * mask_embedding_x + (1 - pi) * embedding_x

        # transformer
        attention_mask = self.gen_attention_masks(segment_ids).to(device)
        feedforward_x = None
        for i in range(self.num_hidden_layers):
            if Debug:
                print('获取第%s个transformer-block %s' % (i, get_time()))
            if i == 0:
                feedforward_x = self.transformer_blocks[i](embedding_i, attention_mask)
            else:
                feedforward_x = self.transformer_blocks[i](feedforward_x, attention_mask)

        # residual connection
        residual_x = embedding_x + feedforward_x

        # mlm
        if Debug:
            print('进行mlm全连接 %s' % get_time())
        output = self.mlm(residual_x)
        return output
