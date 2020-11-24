import torch.nn as nn


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(BiGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bi_gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=True
        )
        self.bi_gru_dense = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.bi_gru_normalization = nn.LayerNorm(self.hidden_size)
        self.bi_gru_dropout = nn.Dropout(p=self.dropout)

    def forward(self, input):
        gru_out, _ = self.bi_gru(input)
        gru_out = self.bi_gru_dense(gru_out)
        gru_out = self.bi_gru_normalization(gru_out)
        gru_out = self.bi_gru_dropout(gru_out)
        return gru_out
