from models import Transformer as Tf

import torch
import torch.nn as nn


class SetGenerator(nn.Module):
    def __init__(self,
                 set_dim: int,
                 latent_dim: int,
                 n_layer: int,
                 head_width: int,
                 encoder: nn.Module,
                 size_predictor: nn.Module,
                 initial_set,
                 n_head: int,
                 ff_width: int,
                 dropout=0.1, weight_sharing=False, normalize=False):
        """
        Args:
            set_dim: dimension of a set
            latent_dim: dimension of the latent space
            n_layer: Number of layer of the transformer
            head_width: hidden size of the transformer attention head
            encoder: encoder used
            size_predictor: network predicting the size of the set after the encoding, only used in test
            initial_set: set initialization method
            n_head: number of attention head
            ff_width: hidden width of the feed forward network
            dropout: dropout rate for the attention
            weight_sharing: shares weights between transformer's layers
            normalize: normalization after residual connections
        """
        super(SetGenerator, self).__init__()

        self.encoder = encoder
        self.size_predictor = size_predictor
        self.initial_set = initial_set
        self.transformer = Tf.Transformer(set_dim + latent_dim, set_dim, n_layer, head_width, n_head, ff_width,
                                          dropout, weight_sharing, normalize)

    def forward(self, x: torch.Tensor, train=True) -> torch.Tensor:
        """
        Args:
            x: Tensor of a set [batch size, size of set, dimension]
            train: If training is activated the size predictor is not used
        """
        batch_size, set_size, features = x.size()
        encoded = self.encoder(x)
        if train:
            size = set_size
        else:
            size = self.size_predictor(encoded)

        init_set = self.initial_set(encoded, size)

        latent_features = encoded.size()[-1]
        encoded = encoded.repeat(1, 1, size)
        encoded = encoded.view(batch_size, size, latent_features)
        transformer_input = torch.cat((init_set, encoded), dim=2)

        output = self.transformer(transformer_input)
        output = output
        return output
