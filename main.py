from SetGenerator import set_translation
from load_config import Configuration
from models import DeepSets as Ds
from models import TransformerSet as Sg
from models.Loss import ChamferLoss
from test import RandSetMLP, RandSet

import Set as S
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import torch


parser = argparse.ArgumentParser()

parser.add_argument('epoch', metavar='e', type=int, help='Number of epoch')
parser.add_argument('batch_size', metavar='bs', type=int, help='Size of a batch')
parser.add_argument('learning_rate', metavar='lr', type=int, help='Negative exponent of the learning rate')

args = parser.parse_args()

cfg = Configuration("config.yml")

encoder = Ds.DeepSetEncoder(cfg.set_n_feature, cfg.latent_dimension, cfg.encoder_layer, cfg.encoder_width)


predictor = Ds.MLP(cfg.latent_dimension, 1, cfg.encoder_width, cfg.encoder_layer)

# ran_init = RandSetMLP(cfg.latent_dimension, cfg.set_n_feature, cfg.set_n_points)
ran_init = RandSet(cfg.set_n_feature)
net = Sg.SetGenerator(cfg.set_n_feature, cfg.latent_dimension, cfg.transformer_layer, cfg.attention_width,
                      encoder, predictor, ran_init, cfg.attention_heads, cfg.feed_forward_width,
                      cfg.attention_dropout, cfg.transformer_sharing, cfg.transformer_norm).double()

sets = np.array([S.RecTriangle().set for i in range(cfg.dataset_size)])
# sets = set_translation(cfg.dataset_size, 1, [1, 0], cfg.set_n_points)
train_set = torch.from_numpy(sets)

criterion = ChamferLoss()
optimizer = optim.Adam(net.parameters(), lr=10**-args.learning_rate)
loss_stat = []
for e in range(args.epoch):
    batches = train_set.split(args.batch_size, dim=0)
    running_loss = 0

    for i, batch in enumerate(batches):

        optimizer.zero_grad()

        outputs = net(batch)
        loss = criterion(outputs, batch)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1 == 0:
            loss_stat.append(running_loss)
            print(e, running_loss)
            running_loss = 0

S.plot2d(np.array(train_set.tolist()[0]))
S.plot2d(np.array(net((train_set[0:1])).tolist()[0]))
S.plt.axis("scaled")
S.plt.show()
