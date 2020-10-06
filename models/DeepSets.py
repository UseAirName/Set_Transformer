from models.Layers import MLP

import torch
import torch.nn as nn


class DeepSetEquivariant(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias=True):
        super(DeepSetEquivariant, self).__init__()
        self.lin_transform = nn.Linear(dim_in, dim_out, bias)
        self.lin_transmit = nn.Linear(dim_in, dim_out, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of a set : [Batch size, size_set, dimension]
        Returns:
            x(A + 11^T B) where A and B are learnable matrices
        """
        bs, n, d = x.size()
        lin1 = self.lin_transform(x)
        lin2 = torch.ones(n, n) @ self.lin_transmit(x)
        return lin1 + 1/n * lin2


class DeepSetInvariant(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        """
        Args:
            phi: A network with an input dimension equals to the number of feature
            rho: A network with an input dimension equals to the output dimension of phi
        """
        super(DeepSetInvariant, self).__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of a set : [Batch size, size_set, dimension]
        Returns:
            rho( sum_i ( phi(x_i) )
        """
        out_phi = self.phi(x)
        summed = torch.sum(out_phi, dim=1)
        out = self.rho(summed)
        return out


class DeepSetEncoder(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, nb_layer: int, width: int):
        """
        Args:
            dim_in: Input dimension
            dim_out: Output dimension
            nb_layer: number of layers for rho and phi networks
            width: width of the rho and phi networks
        """
        super(DeepSetEncoder, self).__init__()
        self.phi = MLP(dim_in, dim_in, width, nb_layer)
        self.rho = MLP(dim_in, dim_out, width, nb_layer)
        self.encoder = DeepSetInvariant(self.phi, self.rho)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of a set [batch size, size of set, dimension]
        """
        out = self.encoder(x)
        return out
