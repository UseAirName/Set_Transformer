from models import DeepSets as Ds

import torch
import torch.nn as nn


class EncDec(nn.Module):
    def __init__(self, features, phi_layer, rho_layer, phi_width, rho_width,
                 size, width, n_layer):
        super(EncDec, self).__init__()
        self.encoder = Ds.DeepSetEncoder(features, phi_layer, rho_layer, phi_width, rho_width)
        self.decoder = Ds.MLP(features, size * features, width, n_layer)

    def forward(self, x):
        enc = self.encoder(x)
        out = self.decoder(enc)
        out = out.view(out.size()[0], x.size()[1], -1)
        return out


class RandSetMLP(nn.Module):
    def __init__(self, latent_features, set_features, size):
        super(RandSetMLP, self).__init__()
        self.MLP = Ds.MLP(latent_features, set_features * size, 32, 3)

    def forward(self, x, set_size):
        out = self.MLP(x)
        return out.view(x.size()[0], set_size, -1)


class RandSet(nn.Module):
    def __init__(self, n_features):
        super(RandSet, self).__init__()
        self.n_features = n_features

    def forward(self, x, set_size):
        bs, l_features = x.size()
        out = torch.randn(bs, set_size, self.n_features, dtype=torch.double)
        return out


class Baseline(nn.Module):
    def __init__(self, n_feature, n_layer, width, size, encoder):
        super(Baseline, self).__init__()
        self.encoder = encoder
        self.MLP = Ds.MLP(n_feature, size*n_feature, width, n_layer)

    def forward(self, x):
        z = self.encoder(x)
        out = self.MLP(z)
        return out.view(x.size())

