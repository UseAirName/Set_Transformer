from models.MultiHeadAttention import MultiHeadAttention
from models.DeepSets import MLP

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_feature, attention_width, forward_width,
                 forward_layer, forward_stride=0, n_head=6, dropout=0.1, bias=True):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(n_feature, attention_width, n_head, dropout, bias)
        self.ff = MLP(n_feature, n_feature, forward_width, forward_layer, forward_stride, bias)

    def forward(self, x):
        """
        Args:
            x: Tensor representing a set : [Batch size, point per set, features per point]

        Returns:
            The forward pass in an Transformer encoder layer
        """
        residual = x
        attention = self.mha(x, x, x)
        attention = attention + residual
        norm = F.normalize(attention)
        residual = norm
        fed = self.ff(norm)
        out = F.normalize(fed + residual)
        return out


class Transformer(nn.Module):
    def __init__(self, n_feature,
                 nb_encoder,
                 attention_width,
                 forward_width,
                 forward_layer=4, forward_stride=0, n_head=6, dropout=0.1, bias=True):
        self.encoders = nn.ModuleList()
        for i in range(nb_encoder):
            self.encoders.append(TransformerEncoderLayer(n_feature, attention_width, forward_width,
                                                         forward_layer, forward_stride, n_head, dropout, bias))
        self.lin = nn.Linear(n_feature, n_feature, bias)

    def forward(self, x):
        for layer in self.encoders:
            x = layer(x)
        out = self.lin(x)
        return out


