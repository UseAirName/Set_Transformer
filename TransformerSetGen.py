import DeepSets as Ds
from MultiHeadAttention import MultiHeadAttention

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_feature, hidden_size, n_head=6, dropout=0.1, bias=True):
        super(TransformerEncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(n_feature, hidden_size, n_head, dropout, bias)

    def forward(self, x):
        attention = self.mha(x)
        added = torch.add(attention, x)
        norm = F.normalize(added)
        return norm
