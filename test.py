from models import DeepSets as Ds
from models import Transformer as Tsg
from models import TransformerSet as Ts
import SetGenerator as Sg
import Loss as Lo

import torch
import torch.nn as nn
import torch.optim as optim


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


#    net = EncDec(nb_features, 3, 3, 30, 30, size_set, 30, 4).double()
#    s = sg.set_translation(nb_set, 2, [1, 0], size_set)
#    s = torch.from_numpy(s)
#    criterion = nn.MSELoss()
#    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


class RandSetMLP(nn.Module):
    def __init__(self, n_features, size):
        super(RandSetMLP, self).__init__()
        self.MLP = Ds.MLP(n_features, n_features * size, 128, 4)

    def forward(self, x, set_size):
        out = self.MLP(x)
        return out.view(x.size()[0], set_size, -1)


class RandSet(nn.Module):
    def __init__(self):
        super(RandSet, self).__init__()

    def forward(self, x, set_size):
        bs, features = x.size()
        out = torch.randn(bs, set_size, features, dtype=torch.double)
        return out


if __name__ == '__main__':
    size_set = 10
    nb_set = 8192
    nb_features = 2
    nb_layer = 3
    hidden_width = 128
    batch_size = 32
    epoch = 2
    learning_rate = 10**(-3)

#    rand_set_init = RandSet().double()
    rand_set_init = RandSetMLP(nb_features, size_set).double()
    encoder = Ds.DeepSetEncoder(nb_features, nb_layer, nb_layer, hidden_width, hidden_width).double()
    size_predictor = Ds.MLP(nb_features, 1, hidden_width, nb_layer).double()
    set_gen = Ts.SetGenerator(nb_features, nb_layer, hidden_width, hidden_width, nb_features,
                              encoder, size_predictor, rand_set_init)
    set_gen = set_gen.double()

    sets = Sg.set_translation(nb_set, 5, [1, 0], size_set)
    data_set = torch.from_numpy(sets)
    train_set, test_set = data_set.split([nb_set//2, nb_set//2], dim=0)

    criterion = Lo.ChamferLoss()
    optimizer = optim.SGD(set_gen.parameters(), lr=learning_rate, momentum=0.9)

    for e in range(epoch):
        batches = train_set.split(batch_size, dim=0)
        for batch in batches:

            optimizer.zero_grad()

            outputs = set_gen(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()

            # print statistics
            print(e, loss.item())









