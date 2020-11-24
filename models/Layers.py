import torch
import torch.nn as nn
import torch.nn.functional as f

# Various Layers to build networks or initialize set


class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, width: int, nb_layers: int, skip=1, bias=False):
        """
        Args:
            dim_in: input dimension
            dim_out: output dimension
            width: hidden width
            nb_layers: number of layers
            skip: jump from residual connections
            bias: indicates presence of bias
        """
        super(MLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.width = width
        self.nb_layers = nb_layers
        self.hidden = nn.ModuleList()
        self.lin1 = nn.Linear(self.dim_in, width, bias)
        self.skip = skip
        self.residual_start = dim_in == width
        self.residual_end = dim_out == width
        for i in range(nb_layers-2):
            self.hidden.append(nn.Linear(width, width, bias))
        self.lin2 = nn.Linear(width, dim_out, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: a tensor with last dimension equals to dim_in
        """
        out = self.lin1(x)
        if self.residual_start:
            out += x
        for layer in self.hidden:
            out = out + layer(f.relu(out))
        out = self.lin2(f.relu(out))
        if self.residual_end:
            out += out
        return out


class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: a tensor
        Returns:
            The sum of the tensor on the first dimension
        """
        summed = torch.sum(x, dim=1)
        return summed


class RandSetMLP(nn.Module):
    # Create an initial set from the latent vector from an MLP
    def __init__(self, latent_features, set_features, size, width):
        super(RandSetMLP, self).__init__()
        self.size = size
        self.MLP = MLP(latent_features, set_features * size, width, 3)

    def forward(self, x):
        out = self.MLP(x)
        return out.view(x.size()[0], self.size, -1)


class RandSet(nn.Module):
    # Create an initial set randomly
    def __init__(self, n_features, size):
        super(RandSet, self).__init__()
        self.n_features = n_features
        self.size = size

    def forward(self, x):
        out = torch.randn(x.size()[0], self.size, self.n_features, dtype=torch.double)
        return out
