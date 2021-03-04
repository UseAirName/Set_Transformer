from load_config import Configuration
from models.Loss import HungarianLoss, HungarianVAELoss, ChamferLoss, ChamferVAELoss, chamfer_loss
from models.CustomNet import CustomVAE
from Set import RecTriangle, CovMatrixSet, triangle_score, matrix_score, plot2d, plot_t
from mpl_toolkits.mplot3d import Axes3D
from MNIST_set import MNISTSet

import matplotlib.pyplot as plt
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

parser.add_argument('--gpu', type=int, help='Id of gpu device. By default use cpu')
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
    config.latent_dim = cfg.latent_dimension
    if cfg.baseline:
        config.baseline_layers = cfg.baseline.encoder_layer + cfg.baseline.decoder_layer
        config.baseline_dim = cfg.baseline.encoder_dim + cfg.baseline.decoder_dim
    config.dataset_size = cfg.dataset_size
    config.set_size = cfg.set_n_points
    config.set_dim = cfg.set_n_feature
    config.loss = cfg.criterion
    config.type_set = cfg.data_type

# Choice of the loss
if cfg.criterion == "HungarianVAE":
    criterion = HungarianVAELoss()
elif cfg.criterion == "ChamferVAE":
    criterion = ChamferVAELoss()

# Use on GPU
use_cuda = args.gpu is not None and torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
else:
    device = "cpu"
args.device = device
args.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print('Device used:', device)


# Loading MNIST dataset
MNIST_train = MNISTSet(train=True, max_size=cfg.set_n_points, path=path_dataset)
MNIST_test = MNISTSet(train=False, max_size=cfg.set_n_points, path=path_dataset)
dataset_train = MNIST_train.dataset.float()
dataset_test = MNIST_test.dataset.float()
labels_train = MNIST_train.labels

# Building the networks from the configuration file
net = CustomVAE(cfg, cfg.network).float()
if cfg.baseline_on:
    baseline = CustomVAE(cfg, cfg.baseline).float()
net.to(device)


def fit(network: nn.Module, dataset: torch.Tensor, epoch: int, batch_size: int, learning_rate: float, crt: nn.Module):
    optimizer = opt.Adam(net.parameters(), lr=learning_rate, weight_decay=10**-6)
    count = 0
    print_interval = 1000
    running_loss = 0
    for e in range(epoch):
        batches = dataset.split(batch_size, dim=0)
        for data in batches:
            data.to(device)
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
    latent_vectors = torch.normal(0, 1, size=(set_size, cfg.latent_dimension)).float()
    out = decoder(latent_vectors)
    return out


def plot2d(x):
    """
        Plot a tensor representing a point_cloud with random color
    """
    plt.scatter(x[:, 0], x[:, 1], color=np.random.rand(3, ), s=12)


fit(dataset_train, args.epoch, args.batch_size, args.lr)
generated_set = generate(1000)


# def plot_t(x):
#     """
#         Plot a tensor representing a point_cloud with random color
#     """
#     return pa.Polygon(x, color=np.random.rand(3, ), fill=False)


if cfg.criterion == "Chamfer":
    criterion = ChamferLoss()
elif cfg.criterion == "Hungarian":
    criterion = HungarianLoss()
elif cfg.criterion == "HungarianVAE":
    criterion = HungarianVAELoss()
elif cfg.criterion == "ChamferVAE":
    criterion = ChamferVAELoss()

fit(net, dataset_train, args.epoch, args.batch_size, 10**-args.learning_rate, criterion)
output, mean, log_var = net(dataset_test)
loss = ChamferLoss()(output, dataset_test)
wandb.log({"loss": loss})


generated_set = generate(net, 21)
for i in range(20):
    plot2d(generated_set[i].detach())
    plt.xlabel('')
    wandb.log({'gen'+str(i): plt})
    plt.clf()

colors = ['black', 'grey', 'brown', 'yellow', 'orange', 'red', 'pink', 'purple', 'blue', 'green']
#

mean_out, var_out = net.encoder(dataset_train[:1000])
vector_out = net.reparameterize(mean_out, var_out)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, v in enumerate(vector_out):
    x, y, z = v.detach()
    ax.scatter(x.numpy(), y.numpy(), z.numpy(), c=colors[labels_train[i]])
wandb.log({'labels_repartition': wandb.Image(plt)})

