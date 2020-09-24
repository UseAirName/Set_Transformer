import SetGenerator as sg
import DeepSets as ds

import torch
import torch.nn as nn
import torch.optim as optim

size_set = 30
nb_set = 1000
nb_features = 2


class enc_dec(nn.Module):
    def __init__(self, features, phi_layer, rho_layer, phi_width, rho_width,
                 size, width, n_layer):
        super(enc_dec, self).__init__()
        self.encoder = ds.DeepSetEncoder(features, phi_layer, rho_layer, phi_width, rho_width)
        self.decoder = ds.MLP(features, size * features, width, n_layer)

    def forward(self, x):
        enc = self.encoder(x)
        out = self.decoder(enc)
        out = out.view(out.size()[0],x.size()[1],-1)
        return out


net = enc_dec(nb_features, 3, 3, 30, 30, size_set, 30, 4).double()
x = sg.set_translation(nb_set, 2, [1, 0], size_set)
x = torch.from_numpy(x)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


