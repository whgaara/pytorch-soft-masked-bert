import torch.nn as nn

from smbert.layers.FeedForward import FeedForward
from smbert.layers.MultiHeadSelfAttention import MultiHeadSelfAttention


class Transformer(nn.Module):
    def __init__(self,
                 hidden_size,
                 attention_head_num,
                 attention_head_size,
                 intermediate_size,
                 dropout_prob=0.1
                 ):
        super(Transformer, self).__init__()
        self.multi_attention = MultiHeadSelfAttention(
            attention_head_num=attention_head_num,
            attention_head_size=attention_head_size)
        self.attention_layernorm = nn.LayerNorm(hidden_size)
        self.feedforward = FeedForward(
                hidden_size,
                intermediate_size,
                dropout_prob)
        self.feedforward_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x, attention_mask):
        attention_x = self.multi_attention(x, attention_mask)
        attention_x = x + attention_x
        attention_x = self.attention_layernorm(attention_x)

        feedforward_x = self.feedforward(attention_x)
        feedforward_x = attention_x + feedforward_x
        feedforward_x = self.feedforward_layernorm(feedforward_x)

        return feedforward_x
