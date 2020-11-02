import torch.nn as nn

from smbert.layers.Gelu import GELU


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super(FeedForward, self).__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.feedforward_act = GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, attention_x):
        attention_x = self.dense1(attention_x)
        attention_x = self.feedforward_act(attention_x)
        attention_x = self.dense2(attention_x)
        attention_x = self.dropout(attention_x)
        return attention_x
