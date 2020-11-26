from load_config import Configuration, NetworkCfg
from models.MultiHeadAttention import MultiHeadAttention
from models.Layers import MLP, Sum

import torch
import torch.nn as nn


class CustomEncoder(nn.Module):
    def __init__(self, cfg: Configuration, net: NetworkCfg):
        super(CustomEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList()
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
        # Layer to learn the mean
        self.mu_layer = nn.Linear(cfg.latent_dimension, cfg.latent_dimension)
        # Layer to learn the log_var
        self.log_var_layer = nn.Linear(cfg.latent_dimension, cfg.latent_dimension)

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        # z dim is [b, latent_dimension]
        z = torch.zeros(x.size(0), self.latent_dim).float()
        layer_nb = 0
        for layer in self.encoder_layers:
            # out dim is [bs, n, d]
            out = layer(x)
            if self.residuals and layer_nb:
                # x dim is [bs, n, d]
                x = out + x
            else:
                x = out
            layer_nb += 1
            # pools dim are [bs, d]
            avg_pool = nn.AvgPool1d(x.size(1))(x.transpose(1, 2)).transpose(1, 2).squeeze()
            max_pool = nn.MaxPool1d(x.size(1))(x.transpose(1, 2)).transpose(1, 2).squeeze()
            z = z + torch.cat((avg_pool, max_pool), dim=1)
        return self.mu_layer(z), self.log_var_layer(z)


class CustomDecoder(nn.Module):
    def __init__(self, cfg: Configuration, net: NetworkCfg):
        super(CustomDecoder, self).__init__()
        self.initial_set = nn.Parameter(torch.randn(cfg.set_n_points, cfg.set_n_feature).float(), requires_grad=True)
        self.size = cfg.set_n_points
        self.mlp_z = []
        self.mlp_o = []
        self.mlp_s = []
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
            self.mlp_z.append(MLP(dim[-1] * cfg.set_n_points, cfg.latent_dimension, 32, 2))
            self.mlp_s.append(MLP(cfg.latent_dimension, cfg.set_n_points * cfg.set_n_feature, 32, 2))
            self.mlp_o.append(MLP(cfg.set_n_feature, dim[-1], 32, 2))
        self.ml_z = MLP(cfg.latent_dimension, cfg.set_n_points * cfg.set_n_feature, net.decoder_dim[0][-1], 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # z dim is [bs, latent_dim]
        z = self.ml_z(x).view(x.size(0), self.size, -1)
        out = z + self.initial_set.repeat((x.size(0), 1, 1))
        z = x
        s = out
        for layer, mlpz, mlps, mlpo in zip(self.decoder_layers, self.mlp_z, self.mlp_s, self.mlp_o):
            # out dim is [bs, n, d]
            out_l = layer(out)
            out = out_l + (out if self.residuals else torch.zeros_like(out_l).float())
            # pools dim are [bs, d]
            max_pool = nn.MaxPool1d(out_l.size(1))(x.transpose(1, 2)).transpose(1, 2).squeeze()
            avg_pool = nn.AvgPool1d(out_l.size(1))(x.transpose(1, 2)).transpose(1, 2).squeeze()
            # z dim is [bs ,d]
            z = z + torch.cat((avg_pool, max_pool), dim=1)
            # ! update of s not good yet !
            s = s + mlps(z).view_as(s)
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
