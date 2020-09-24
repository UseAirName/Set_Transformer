import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    # MultiHeadAttention Based on "The Illustrated Transformer"
    def __init__(self, n_feature, hidden_size, n_head=6, dropout=0.1, bias=True):
        """
        Args:
            n_feature: Number of feature in the set
            hidden_size: width of a head
            n_head: number of head
            dropout: dropout rate
            bias: boolean for bias
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.bias = bias
        self.n_feature = n_feature

        self.q_lin = nn.Linear(self.n_feature, self.hidden_size * self.n_head, self.bias)
        self.k_lin = nn.Linear(self.n_feature, self.hidden_size * self.n_head, self.bias)
        self.v_lin = nn.Linear(self.n_feature, self.hidden_size * self.n_head, self.bias)

        self.dropout_layer = nn.Dropout(dropout)

        self.out_layer = nn.Linear(self.hidden_size * self.n_head, self.n_feature, bias)

    def forward(self, q, k, v):
        """
        Args:
            q: Query tensor : [Batch size, size of the set, features per point]
            k: Key   tensor : [Batch size, size of the set, features per point]
            v: Value tensor : [Batch size, size of the set, features per point]
        """
        batch = q.size()[0]
        p_per_set = q.size()[1]

        v = self.v_lin(v).view(batch, p_per_set, self.n_head, self.hidden_size).transpose(1, 2)
        q = self.q_lin(q).view(batch, p_per_set, self.n_head, self.hidden_size).transpose(1, 2)
        k = self.k_lin(k).view(batch, p_per_set, self.n_head, self.hidden_size).transpose(1, 2).transpose(2, 3)

        score = torch.matmul(q, k.transpose(-2, -1))
        score.mul_(1 / np.sqrt(self.hidden_size))
        soft_score = F.softmax(score, dim=-1)
        soft_score = self.dropout_layer(soft_score)

        attention = torch.matmul(soft_score, v)
        attention = attention.transpose(1, 2)
        attention = attention.reshape(batch, p_per_set, self.hidden_size * self.n_head)

        out = self.out_layer(attention)
        return out
