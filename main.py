from load_config import Configuration
from models.Loss import HungarianLoss, HungarianVAELoss, ChamferLoss, ChamferVAELoss
from models.CustomNet import CustomVAE
from Set import RecTriangle, CovMatrixSet, triangle_score, matrix_score
from sklearn.datasets import make_spd_matrix
from MNIST_set import MNISTSet

import matplotlib.pyplot as plt
import matplotlib.patches as pa
import torch.optim as opt
import torch.nn as nn
import numpy as np
import argparse
import wandb
import torch
import os

# Setting the paths if used on server
if os.path.isdir('/SCRATCH2'):
    path_wandb = '/SCRATCH2/tbordin/'
else:
    path_wandb = './'

if os.path.isdir('/dataset/'):
    path_dataset = '/dataset/MNIST'
else:
    path_dataset = './MNIST'

parser = argparse.ArgumentParser()

# Parsing the number of epoch, batch size, and learning rate
parser.add_argument('--epoch', type=int, help='Number of epoch', default=300, required=False)
parser.add_argument('--batch_size', type=int, help='Size of a batch', default=16, required=False)
parser.add_argument('--lr', type=float, default=5e-4, required=False)

args = parser.parse_args()

# Reading the configuration file
cfg = Configuration("config.yml")

# saving the configuration in wandb
if cfg.wandb_on:
    wandb.init(entity="tbordin", project="EPFL", dir=path_wandb)

    config = wandb.config
    config.batch_size = args.batch_size
    config.epoch = args.epoch
    config.lr = args.learning_rate
    config.encoder = cfg.network.encoder_layer
    config.encoder_dim = cfg.network.encoder_dim
    config.decoder = cfg.network.decoder_layer
    config.decoder_dim = cfg.network.decoder_dim
    if cfg.baseline:
        config.baseline_layers = cfg.baseline.encoder_layer + cfg.baseline.decoder_layer
        config.baseline_dim = cfg.baseline.encoder_dim + cfg.baseline.decoder_dim
    config.dataset_size = cfg.dataset_size
    config.set_size = cfg.set_n_points
    config.set_dim = cfg.set_n_feature
    config.loss = cfg.criterion
    config.type_set = cfg.data_type

# Building the network from the configuration file
net = CustomVAE(cfg, cfg.network).double()

if cfg.baseline_on:
    baseline = CustomVAE(cfg, cfg.baseline).double()

# Loading MNIST dataset
dataset_train = MNISTSet(train=True, max_size=cfg.set_n_points, path=path_dataset).dataset.double()
# dataset_test = MNISTSet(train=False, max_size=cfg.set_n_points, path=path_dataset).dataset.double()


def fit(network: nn.Module, dataset: torch.Tensor, epoch: int, batch_size: int, learning_rate: float, crt: nn.Module):
    optimizer = opt.Adam(net.parameters(), lr=learning_rate)
    count = 0
    print_interval = 1000
    running_loss = 0
    for e in range(epoch):
        batches = dataset.split(batch_size, dim=0)
        for data in batches:
            count += 1
            optimizer.zero_grad()

            outputs, mean, log_var = network(data)
            loss = crt(outputs, mean, log_var, data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if count % print_interval == print_interval - 1:
                print("epoch: ", e, ", loss: ", running_loss / print_interval)
                running_loss = 0


def generate(network, set_size):
    decoder = network.decoder
    latent_vectors = torch.normal(0, 1, size=(set_size, cfg.latent_dimension)).double()
    out = decoder(latent_vectors)
    return out


def plot2d(x):
    """
        Plot a tensor representing a point_cloud with random color
    """
    plt.scatter(x[:, 0], x[:, 1], color=np.random.rand(3, ), s=12)


fit(dataset_train, args.epoch, args.batch_size, args.lr)
generated_set = generate(1000)


def plot_t(x):
    """
        Plot a tensor representing a point_cloud with random color
    """
    return pa.Polygon(x, color=np.random.rand(3, ), fill=False)


if cfg.criterion == "Chamfer":
    criterion = ChamferLoss()
elif cfg.criterion == "Hungarian":
    criterion = HungarianLoss()
elif cfg.criterion == "HungarianVAE":
    criterion = HungarianVAELoss()
elif cfg.criterion == "ChamferVAE":
    criterion = ChamferVAELoss()

fit(net, dataset_train, args.epoch, args.batch_size, 10**-args.learning_rate, criterion)
output = net(dataset_train[:30])

# Saving 20 pair of image and results at the end of the run
for i in range(20):
    plot2d(dataset_train[i].detach())
    plt.xlabel('dummy')
    wandb.log({'fig'+str(i): plt})
    plt.clf()
    plot2d(output[0][i].detach())
    plt.xlabel('dummy')
    wandb.log({'rec'+str(i): plt})
    plt.clf()

# generated_set = generate(net, 1000)
# score1 = matrix_score(generated_set, cov_matrix)
# score2 = matrix_score(train_set, cov_matrix)


# if cfg.wandb_on:
#     plt.hist(score1, bins=100)
#     plt.xlabel("Number of set")
#     plt.ylabel("MSE value")
#     wandb.log({"hist_gen": plt})
#     plt.clf()
#     plt.hist(score2, bins=100)
#     plt.xlabel("Number of set")
#     plt.ylabel("MSE value")
#     wandb.log({"hist_train": plt})
#
# if cfg.baseline_on:
#     wandb.watch(baseline)
#     fit(baseline, train_set, args.epoch, args.batch_size, 10 ** -args.learning_rate, criterion)
#     generated_set = generate(baseline, 1000)
#     score3 = matrix_score(generated_set, cov_matrix)
#     plt.clf()
#     plt.hist(score3, bins=100)
#     plt.xlabel("Number of set")
#     plt.ylabel("MSE value")
#     wandb.log({"hist_baseline": plt})
