from load_config import Configuration, NetworkCfg
from models.MultiHeadAttention import MultiHeadAttention
from models.Layers import MLP, Sum

import torch
import torch.nn as nn


class CustomEncoder(nn.Module):
    def __init__(self, cfg: Configuration, net: NetworkCfg):
        super(CustomEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.linear_o = []
        self.linear_z = []
        self.residuals = cfg.residuals
        self.latent_dim = cfg.latent_dimension
        for layer, dim in zip(net.encoder_layer, net.encoder_dim):
            if layer == "MultiHeadAttention":
                self.encoder_layers.append(MultiHeadAttention(dim[0], cfg.head_width, cfg.n_head))
            if layer == "Linear":
                self.encoder_layers.append(nn.Linear(dim[0], dim[1], bias=True))
            if layer == "Sum":
                self.encoder_layers.append(Sum())
            if layer == "MLP":
                self.encoder_layers.append(MLP(dim[0], dim[-1], dim[1], len(dim)-2))
            # Linear layer to update z and cast to the right dimension
            self.linear_z.append(nn.Linear(dim[-1] * cfg.set_n_points, cfg.latent_dimension))
            self.linear_o.append(nn.Linear(cfg.latent_dimension, dim[-1] * cfg.set_n_points))
        # Layer to learn the mean
        self.mu_layer = nn.Linear(cfg.latent_dimension, cfg.latent_dimension)
        # Layer to learn the log_var
        self.log_var_layer = nn.Linear(cfg.latent_dimension, cfg.latent_dimension)

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        z = torch.zeros(x.size(0), self.latent_dim).float()
        for layer, lino, linz in zip(self.encoder_layers, self.linear_o, self.linear_z):
            out = layer(x)
            if self.residuals:
                x = out + lino(z).view_as(out)
            else:
                x = out
            z = z + linz(out.view(out.size(0), -1))
        return self.mu_layer(z), self.log_var_layer(z)


class CustomDecoder(nn.Module):
    def __init__(self, cfg: Configuration, net: NetworkCfg):
        super(CustomDecoder, self).__init__()
        self.initial_set = nn.Parameter(torch.randn(cfg.set_n_points, cfg.set_n_feature).float(), requires_grad=True)
        self.size = cfg.set_n_points
        self.linear_z = []
        self.linear_o = []
        self.linear_s = []
        self.decoder_layers = nn.ModuleList()
        self.residuals = cfg.residuals
        for layer, dim in zip(net.decoder_layer, net.decoder_dim):
            if layer == "MultiHeadAttention":
                self.decoder_layers.append(MultiHeadAttention(dim[0], cfg.head_width, cfg.n_head))
            if layer == "Linear":
                self.decoder_layers.append(nn.Linear(dim[0], dim[-1], bias=True))
            if layer == "Sum":
                self.decoder_layers.append(Sum())
            if layer == "MLP":
                self.decoder_layers.append(MLP(dim[0], dim[-1], dim[1], len(dim)))
            # Linear layer to update z and s and cast to the right dimension
            self.linear_z.append(nn.Linear(dim[-1] * cfg.set_n_points, cfg.latent_dimension))
            self.linear_s.append(nn.Linear(cfg.latent_dimension, cfg.set_n_points * cfg.set_n_feature))
            self.linear_o.append(nn.Linear(cfg.set_n_feature, dim[-1]))
        self.mlp_z = MLP(cfg.latent_dimension, cfg.set_n_points * cfg.set_n_feature, net.decoder_dim[0][-1], 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.mlp_z(x).view(x.size(0), self.size, -1)
        out = z + self.initial_set.repeat((x.size(0), 1, 1))
        z = x
        s = out
        for layer, linz, lins, lino in zip(self.decoder_layers, self.linear_z, self.linear_s, self.linear_o):
            out_l = layer(out)
            z = z + linz(out_l.view(out.size(0), -1))
            out = out_l + (lino(s) if self.residuals else torch.zeros_like(out_l).float())
            s = s + lins(z).view_as(s)
        return s


class CustomVAE(nn.Module):
    def __init__(self, cfg: Configuration, net: NetworkCfg):
        super(CustomVAE, self).__init__()
        self.encoder = CustomEncoder(cfg, net)
        self.decoder = CustomDecoder(cfg, net)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu: mean of the encoder's latent space
            log_var: log variance of the encoder's latent space
        """
        std = torch.exp(0.5*log_var)
        z = mu + torch.randn_like(std) * std * 0
        return z

    def forward(self, x):
        latent_mean, log_var = self.encoder(x)
        latent_vector = self.reparameterize(latent_mean, log_var)
        out = self.decoder(latent_vector)
        return out, latent_mean, log_var
