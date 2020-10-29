from load_config import Configuration
from models.Loss import HungarianLoss, HungarianVAELoss
from models.CustomNet import CustomVAE
from Set import RecTriangle, CovMatrixSet, triangle_score

import matplotlib.pyplot as plt
import torch.optim as opt
import numpy as np
import argparse
import torch


parser = argparse.ArgumentParser()

# Parsing the number of epoch, batch size, and learning rate
parser.add_argument('--epoch', type=int, help='Number of epoch', default=300, required=False)
parser.add_argument('--batch_size', type=int, help='Size of a batch', default=16, required=False)
parser.add_argument('--lr', type=float, default=5e-4, required=False)

args = parser.parse_args()

# Reading the configuration file
cfg = Configuration("config.yml")

# Building the network from the configuration file
net = CustomVAE(cfg).double()

# TODO : add the type of the data set in the configuration file
sets = np.array([RecTriangle().set for i in range(cfg.dataset_size)])
train_set = torch.from_numpy(sets)


def fit(dataset: torch.Tensor, epoch: int, batch_size: int, lr: float):
    # TODO : add the loss in the configuration file
    criterion = HungarianVAELoss()
    optimizer = opt.Adam(net.parameters(), lr=lr)
    count = 0
    print_interval = 100
    running_loss = 0
    for e in range(epoch):
        batches = dataset.split(batch_size, dim=0)
        for data in batches:
            count += 1
            optimizer.zero_grad()

            outputs, mean, log_var = net(data)
            loss = criterion(outputs, mean, log_var, data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if count % print_interval == print_interval - 1:
                print("epoch: ", e, ", loss: ", running_loss / print_interval)
                running_loss = 0


def generate(set_size):
    decoder = net.decoder
    latent_vectors = torch.normal(0, 1, size=(set_size, cfg.latent_dimension)).double()
    out = decoder(latent_vectors)
    return out


def plot2d(x: torch.Tensor):
    """
        Plot a tensor representing a point_cloud with random color
    """
    x = x.numpy()
    plt.scatter(x[:, 0], x[:, 1], color=np.random.rand(3, ), s=12)


fit(train_set, args.epoch, args.batch_size, args.lr)
generated_set = generate(1000)

# Plot the histogram of the highest angle
plt.hist(triangle_score(generated_set), 120, color="blue")
# Plot a vertical line at 90 degrees
plt.axvline(90, color="green")
plt.xlabel("Maximum angle value")
plt.ylabel("Number of triangle")
plt.title("Maximum angle for generated triangles")
plt.show()
