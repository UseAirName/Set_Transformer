import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    # MultiHeadAttention Based on "The Illustrated Transformer"
    def __init__(self, dimension: int, hidden_size: int, n_head=6, dropout=0.1):
        """
        Args:
            dimension: dimension of the set
            hidden_size: width of a head
            n_head: number of head
            dropout: dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dim = dimension

        self.q_lin = nn.Linear(self.dim, self.hidden_size * self.n_head)
        self.k_lin = nn.Linear(self.dim, self.hidden_size * self.n_head)
        self.v_lin = nn.Linear(self.dim, self.hidden_size * self.n_head)

        self.dropout_layer = nn.Dropout(dropout)

        self.out_layer = nn.Linear(self.hidden_size * self.n_head, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The considered set [batch size, size of set, dimension]

        Returns:
            Attention computed by the multi-head without any mask
        """
        batch_size = x.size()[0]
        p_per_set = x.size()[1]

        v = self.v_lin(x).view(batch_size, p_per_set, self.n_head, self.hidden_size).transpose(1, 2)
        q = self.q_lin(x).view(batch_size, p_per_set, self.n_head, self.hidden_size).transpose(1, 2)
        k = self.k_lin(x).view(batch_size, p_per_set, self.n_head, self.hidden_size).transpose(1, 2).transpose(2, 3)

        score = torch.matmul(q, k)
        score = score / np.sqrt(self.hidden_size)
        soft_score = F.softmax(score, dim=-1)
        soft_score = self.dropout_layer(soft_score)

        attention = torch.matmul(soft_score, v)
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch_size, p_per_set, self.hidden_size * self.n_head)

        out = self.out_layer(attention)
        return out
