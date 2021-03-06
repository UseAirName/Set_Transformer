from models.MultiHeadAttention import MultiHeadAttention

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dimension: int, hidden_width: int, dropout=0.1):
        """
        Args:
            dimension: dimension of the input
            hidden_width: width of hidden layers
            dropout: dropout rate
        """
        super(FeedForward, self).__init__()
        self.lin1 = nn.Linear(dimension, hidden_width)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_width, dimension)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            Feed forward layer for a transformer
        """
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.lin2(x)
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dimension: int, head_width: int, n_head: int, ff_width: int, normalize=False):
        """
        Args:
            dimension: Dimension of the input
            head_width: Width of an attention head
            n_head: Number of head for the attention computation
            ff_width: Width of hidden layers for the feed forward network
            normalize: Normalization parameter
        """
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(dimension, head_width, n_head)
        self.ff = FeedForward(dimension, ff_width)
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of a set [batch size, size of set, dimension]

        Returns:
            Output of a transformer Encoder Layer
        """
        attention = x + self.mha(x)
        if self.normalize:
            # TODO: Add batch norm here
            attention = attention

        out = attention + self.ff(attention)
        if self.normalize:
            # TODO: Add batch norm here
            out = out

        return out


class Transformer(nn.Module):
    # Multi layer transformer
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 nb_layer: int,
                 head_width: int,
                 n_head: int,
                 ff_width: int,
                 weight_sharing=True, normalize=False):
        """
        Args:
            dim_in: input dimension
            dim_out: output dimension
            nb_layer: number of transformer layer
            head_width: hidden width of attention heads
            n_head: number of head
            ff_width: width of hidden layer of the feed forward network
            weight_sharing: weight sharing between layers
            normalize: normalization parameter
        """
        super(Transformer, self).__init__()
        self.nb_layer = nb_layer
        self.weight_sharing = weight_sharing
        if self.weight_sharing:
            self.layers = TransformerEncoderLayer(dim_in, head_width, n_head, ff_width, normalize)
        else:
            self.layers = nn.ModuleList()
            for i in range(nb_layer):
                self.layers.append(TransformerEncoderLayer(dim_in, head_width, n_head, ff_width))
        self.lin = nn.Linear(dim_in, dim_out, bias=True)
        self.norm = nn.LayerNorm([10, dim_out])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of a set [batch size, number of points, dimension]

        Returns:
            Forward pass through a transformer encoder layer
        """
        if self.weight_sharing:
            for i in range(self.nb_layer):
                x = self.layers(x)
        else:
            for layer in self.layers:
                x = layer(x)
        out = self.lin(x)
        return out
