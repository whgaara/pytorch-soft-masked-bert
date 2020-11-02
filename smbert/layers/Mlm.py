import torch
import torch.nn as nn


class Mlm(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(Mlm, self).__init__()
        self.mlm_dense = nn.Linear(hidden_size, vocab_size)

    def forward(self, feedforward_x):
        feedforward_x = self.mlm_dense(feedforward_x)
        return feedforward_x
