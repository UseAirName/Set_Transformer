import torch
import torch.nn as nn


class DeepSetEquivariant(nn.Module):
    def __init__(self, n_feature, d_out, bias=True):
        super(DeepSetEquivariant, self).__init__()
        self.lin_transform = nn.Linear(n_feature, d_out, bias)
        self.lin_transmit = nn.Linear(n_feature, d_out, bias)

    def forward(self, x):
        """
        Args:
            x: Tensor representing a set: (Batch_size, size_of_set, features per point)

        Returns:
            x(A + 11^T B) where A and B are learnable matrices
        """
        bs, n, d = x.size()
        lin1 = self.lin_transform(x)
        lin2 = torch.ones(n, n) @ self.lin_transmit(x)
        return lin1 + 1/n * lin2


class DeepSetInvariant(nn.Module):
    def __init__(self, phi, rho):
        """
        Args:
            phi: input dimension should be the number of feature
            rho: input dimension is phi's output dimension
        """
        super(DeepSetInvariant, self).__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x):
        """
        Args:
            x: Tensor representing a set: (Batch_size, size_of_set, features per point)

        Returns:
            rho( sum_i ( phi(x_i) )
        """
        points_features = x.view(-1, x.size()[2])
        out_phi = self.phi(points_features)
        out_phi.view(x.size())
        summed = torch.sum(out_phi, dim=1)
        out = self.rho(summed)
        return out


class MLP(nn.Module):
    def __init__(self, d_in, d_out, width, nb_layers, stride=0, bias=True):
        super(MLP, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.width = width
        self.nb_layers = nb_layers
        self.hidden = nn.ModuleList()
        self.lin1 = nn.Linear(self.d_in, width, bias)
        self.stride = stride
        for i in range(nb_layers-1):
            self.hidden.append(nn.Linear(width, width, bias))
        self.lin_last = nn.Linear(width, d_out, bias)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Tensor (Batch_size, d_in)
        """
        out_lin = self.lin1(x)
        residual = out_lin
        for i,layer in enumerate(self.hidden):
            if self.stride != 0:
                if i and i % self.stride == 0:
                    out_lin += residual
            out_lin = layer(out_lin)
            self.hidden.append(nn.ReLU())
        out_lin_last = self.lin_last(out_lin)
        out = self.sig(out_lin_last)
        return out


class DeepSetEncoder(nn.Module):
    def __init__(self, features, phi_layer, rho_layer, phi_width, rho_width):
        """
        Args:
            features: Number of features per point in the set
            phi_layer: Number of layer of phi
            rho_layer: Number of layer of rho
            phi_width: Width of the MLP phi network
            rho_width: Width of the MLP rho network
        """
        super(DeepSetEncoder, self).__init__()
        self.phi = MLP(features, features, phi_width, phi_layer)
        self.rho = MLP(features, features, rho_width, rho_layer)
        self.encoder = DeepSetInvariant(self.phi, self.rho)

    def forward(self, x):
        """
        Args:
            x: Tensor representing a set: (Batch size, size of set, number of feature)
        """
        out = self.encoder(x)
        return out


class DeepSetGenerator(nn.Module):
    def __init__(self):
        super(DeepSetGenerator, self).__init__()
