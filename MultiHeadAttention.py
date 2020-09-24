import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    # MultiHeadAttention Based on "The Illustrated Transformer"
    def __init__(self, n_feature, hidden_size, n_head=6, dropout=0.0, bias=True):
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
        # TODO: docstring with expected size for each tensor
        batch = q.size()[0]

        v = self.v_lin(v).view(batch, self.n_feature, self.n_head, self.hidden_size).transpose(1, 2)
        q = self.q_lin(q).view(batch, self.n_feature, self.n_head, self.hidden_size).transpose(1, 2)
        k = self.k_lin(k).view(batch, self.n_feature, self.n_head, self.hidden_size).transpose(1, 2).transpose(2, 3)

        score = torch.matmul(q, k.transpose(-2, -1))
        score.mul_(1 / np.sqrt(self.hidden_size))
        soft_score = F.softmax(score, dim=-1)
        soft_score = self.dropout_layer(soft_score)

        attention = torch.matmul(soft_score, v)
        attention = attention.transpose(1, 2)
        attention = attention.view(batch, self.n_feature, self.hidden_size * self.n_head)

        out = self.out_layer(attention)
        return out
