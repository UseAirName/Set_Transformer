from load_config import Configuration
from models.MultiHeadAttention import MultiHeadAttention
from models.Layers import MLP, Sum, RandSet, RandSetMLP

import torch
import torch.nn as nn


class CustomEncoder(nn.Module):
    def __init__(self, cfg: Configuration):
        super(CustomEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.residuals_e = []
        for layer, dim in zip(cfg.encoder_layer, cfg.encoder_dim):
            if layer == "MultiHeadAttention":
                self.encoder_layers.append(MultiHeadAttention(dim[0], cfg.head_width, cfg.n_head))
                self.residuals_e.append(True)
            if layer == "Linear":
                self.encoder_layers.append(nn.Linear(dim[0], dim[1], bias=True))
                self.residuals_e.append(False)
            if layer == "Sum":
                self.encoder_layers.append(Sum())
                self.residuals_e.append(False)
            if layer == "MLP":
                self.encoder_layers.append(MLP(dim[0], dim[-1], dim[1], len(dim)-2))
                self.residuals_e.append(False)
        # Layer to learn the mean
        self.mu_layer = nn.Linear(cfg.encoder_dim[-1][-1], cfg.latent_dimension)
        # Layer to learn the log_var
        self.log_var_layer = nn.Linear(cfg.encoder_dim[-1][-1], cfg.latent_dimension)

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        for layer, residual in zip(self.encoder_layers, self.residuals_e):
            if residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return self.mu_layer(x), self.log_var_layer(x)


class CustomDecoder(nn.Module):
    def __init__(self, cfg: Configuration):
        super(CustomDecoder, self).__init__()
        self.initial_set = torch.randn(cfg.set_n_points, cfg.set_n_feature)
        self.size = cfg.set_n_points

        self.decoder_layers = nn.ModuleList()
        self.residuals_d = []
        for layer, dim in zip(cfg.decoder_layer, cfg.decoder_dim):
            if layer == "MultiHeadAttention":
                self.decoder_layers.append(MultiHeadAttention(dim[0], cfg.head_width, cfg.n_head))
                self.residuals_d.append(True)
            if layer == "Linear":
                self.decoder_layers.append(nn.Linear(dim[0], dim[-1], bias=True))
                self.residuals_d.append(False)
            if layer == "Sum":
                self.decoder_layers.append(Sum())
                self.residuals_d.append(False)
            if layer == "MLP":
                self.decoder_layers.append(MLP(dim[0], dim[-1], dim[1], len(dim)-2))
                self.residuals_d.append(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, _ = x.size()
        init_set = self.initial_set.repeat(bs, 1, 1)
        latent_features = x.size()[-1]
        encoded = x.repeat(1, 1, self.size)
        encoded = encoded.view(bs, self.size, latent_features)

        x = torch.cat((init_set, encoded), dim=2)
        for layer, residual in zip(self.decoder_layers, self.residuals_d):
            if residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x


class CustomVAE(nn.Module):
    def __init__(self, cfg: Configuration):
        super(CustomVAE, self).__init__()
        self.encoder = CustomEncoder(cfg)
        self.decoder = CustomDecoder(cfg)

    def reparameterize(self, mu, log_var):
        """
        Args:
            mu: mean of the encoder's latent space
            log_var: log variance of the encoder's latent space
        """
        std = torch.exp(0.5*log_var)
        z = mu + torch.randn_like(std) * std
        return z

    def forward(self, x):
        latent_mean, log_var = self.encoder(x)
        latent_vector = self.reparameterize(latent_mean, log_var)
        out = self.decoder(latent_vector)
        return out, latent_mean, log_var
