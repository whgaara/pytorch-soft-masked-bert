import torch
import torch.nn as nn

from config import device


class SMBbertEmbeddings(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size, dropout_prob=0.1):
        super(SMBbertEmbeddings, self).__init__()
        self.max_len = max_len
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.type_embeddings = nn.Embedding(2, hidden_size)
        self.position_embeddings = nn.Embedding(self.max_len, hidden_size)
        self.emb_normalization = nn.LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(p=dropout_prob)

    def forward(self, input_token, position_ids, segment_ids):
        # mask embeddings
        token_size = input_token.size()
        mask_tensor = torch.LongTensor(token_size).fill_(103).to(device)
        mask_embeddings = self.token_embeddings(mask_tensor)

        # embeddings
        token_embeddings = self.token_embeddings(input_token)
        postion_embeddings = self.position_embeddings(position_ids)
        type_embeddings = self.type_embeddings(segment_ids)

        embedding_x = token_embeddings + type_embeddings + postion_embeddings
        embedding_x = self.emb_normalization(embedding_x)
        embedding_x = self.emb_dropout(embedding_x)

        return embedding_x, mask_embeddings
