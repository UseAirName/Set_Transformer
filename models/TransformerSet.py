from models import DeepSets as Ds
from models import Transformer as Tf

import torch
import torch.nn as nn


class SetGenerator(nn.Module):
    def __init__(self, n_feature,
                 transformer_layer,
                 transformer_hidden_size,
                 transformer_forward_width,
                 set_init_features,
                 encoder,
                 size_predictor,
                 random_set_init,
                 transformer_forward_layer=3, transformer_stride=0, transformer_head=6, dropout=0.1, bias=True):
        """
        Args:
            n_feature: Number of feature per point
            transformer_layer: Number of layer of the transformer
            transformer_hidden_size: Hidden size of the transformer attention head
            transformer_forward_width: Hidden size of the feed forward in the transformer
            set_init_features: number of features at the random set initialization
            encoder: Encoder used
            size_predictor: MLP that predict the size of the set after the encoding, only used in test
            random_set_init: The random set initialization method
            transformer_forward_layer: number of layer in the feed forward of the transformer
            transformer_stride: stride of the forward layer
            transformer_head: number of attention head
            dropout: dropout of the attention
            bias: boolean for presence of bias
        """
        super(SetGenerator, self).__init__()

        self.encoder = encoder
        self.size_predictor = size_predictor
        self.random_set_init = random_set_init
        self.transformer = Tf.Transformer(set_init_features + n_feature, n_feature, transformer_layer,
                                          transformer_hidden_size, transformer_forward_width,
                                          transformer_forward_layer, transformer_stride,
                                          transformer_head, dropout, bias)

    def forward(self, x, train=True):
        """
        Args:
            x: A tensor representing a set [batch_size, size of the set, features per point]
            train: True if we are training the model false otherwise
        """
        batch_size, set_size, features = x.size()
        encoded = self.encoder(x)
        if train:
            size = set_size
        else:
            size = self.size_predictor(encoded)

        init_set = self.random_set_init(encoded, size)

        encoded = encoded.repeat(1, 1, size)
        encoded = encoded.view(batch_size, size, features)
        transformer_input = torch.cat((init_set, encoded), dim=2)

        output = self.transformer(transformer_input)
        return output

