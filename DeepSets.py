import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSetEquivariant(nn.Module):
    def __init__(self, d_in, d_out):
        super(DeepSetEquivariant, self).__init__()
        # TODO: when possible, avoid defining the parameters by yourself
        # Problem 1: they are not initialized in the right way
        # Problem 2: you did not define a function to reinitialize them
        # if you go from d_in to d_out, you can probably use a nn.Linear() instead
        self.w1 = nn.Parameter(torch.randn(d_in, d_out))
        self.w2 = nn.Parameter(torch.randn(d_in, d_out))
        self.bias = nn.Parameter(torch.randn(1, d_out))

    def forward(self, x):
        n, d = x.size()
        lin_transform = torch.matmul(x,self.w1)
        # TODO: instead of matmul, just use @
        out2 = torch.matmul(torch.ones(n, n), x)
        lin_transmit = torch.matmul(out2, self.w2)
        out_bias = torch.matmul(torch.ones(n, 1), self.bias)
        return lin_transform + 1/n * lin_transmit + out_bias


class DeepSetInvariant(nn.Module):
    def __init__(self, phi, rho):
        super(DeepSetInvariant, self).__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x):
        # TODO: you should always put in docstring the expected size for x \
        # Be careful: there will be another dimension corresponding to the batch size, so x is: (bs, n, channels)
        points_features = list(x.split(1, dim=1))
        out_phi = []

        for i in range(len(points_features)):
            # TODO: this would be very slow. You can directly do a batch computation
            # Notice that the MLP can take tensors of any order as input, it will just transform the last dimension
            # and perform the same operation of each entry on the other dimensions
            out_phi.append(self.phi(points_features[i]))
        out_phi = torch.cat(out_phi, dim=1)
        summed = torch.sum(out_phi, dim=1)
        out = self.rho(summed)
        return out


class MLP(nn.Module):
    def __init__(self, d_in, d_out, width, nb_layers):
        super(MLP, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.width = width
        self.nb_layers = nb_layers
        self.hidden = nn.ModuleList()
        self.lin1 = nn.Linear(self.d_in, width)
        for i in range(nb_layers-1):
            self.hidden.append(nn.Linear(width, width))
            # TODO: not a great idea, because the size of the array is not nb_layers - 1. Instead, put the relu in forward
            self.hidden.append(nn.ReLU())
        self.lin_last = nn.Linear(width, d_out)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out_lin = self.lin1(x)
        for layer in self.hidden:
            # TODO: add an option for residual connexions
            out_lin = layer(out_lin)
            # TODO Use F.relu here
        out_lin_last = self.lin_last(out_lin)
        out = self.sig(out_lin_last)
        return out


class DeepSetEncoder(nn.Module):
    def __init__(self, features, phi_layer, rho_layer, phi_width, rho_width):
        super(DeepSetEncoder, self).__init__()
        self.phi = MLP(features, features, phi_width, phi_layer)
        self.rho = MLP(features, features, rho_width, rho_layer)
        self.encoder = DeepSetInvariant(self.phi, self.rho)

    def forward(self, x):
        out = self.encoder(x)
        return out


class DeepSetGenerator(nn.Module):
    def __init__(self):
        super(DeepSetGenerator, self).__init__()


# TODO: put the files Deepset and MultiHeadAttention in a folder `models`