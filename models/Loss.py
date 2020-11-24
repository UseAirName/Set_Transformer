import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from math import exp

# Different Loss function to compare sets


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, set1: torch.Tensor, set2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            set1: Tensor of a set [batch size, point per set, dimension]
            set2: Tensor of a set [batch size, point per set, dimension]
            both dimension must be equal

        Returns:
            The Chamfer distance between both sets
        """

        bs1, n, d1 = set1.size()
        bs2, m, d2 = set2.size()
        assert(d1 == d2 and bs1 == bs2)

        dist = torch.cdist(set1, set2, 2)

        out_dist, _ = torch.min(dist, dim=2)
        out_dist2, _ = torch.min(dist, dim=1)

        total_dist = torch.mean(out_dist) + torch.mean(out_dist2)

        return total_dist/n


class HungarianLoss(nn.Module):
    def __init__(self):
        super(HungarianLoss, self).__init__()

    def forward(self, set1: torch.Tensor, set2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            set1: A tensor of a set [batch_size, number of points, dimension]
            set2: A tensor of a set [batch_size, number of points, dimension]
        Returns:
            The Hungarian distance between set1 and set2
        """
        bs1, n, d1 = set1.size()
        bs2, m, d2 = set2.size()

        assert(d1 == d2 and bs1 == bs2)  # Both sets must have the same dimension to be compared

        batch_dist = torch.cdist(set1, set2, 2)
        numpy_batch_dist = batch_dist.detach().numpy()
        indices = map(linear_sum_assignment, numpy_batch_dist)
        loss = [dist[row_idx, col_idx].sum() for dist, (row_idx, col_idx) in zip(batch_dist, indices)]

        # Mean loss computed on the batch
        total_loss = torch.mean(torch.stack(loss))
        return total_loss/n


class HungarianVAELoss(nn.Module):
    def __init__(self):
        super(HungarianVAELoss, self).__init__()
        self.hungarian = HungarianLoss()

    def forward(self, output: torch.Tensor, mu: torch.Tensor,
                log_var: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output: output of the network
            mu: mean computed in the network
            log_var: log variance computed in the network
            real: expected value to compare to the output
        Returns:
            The variational loss computed as the sum of the hungarian loss and the Kullback-Leiber divergence.
        """
        dkl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        hg = self.hungarian(output, real)
        return 0.0000 * torch.mean(dkl) + hg


class ChamferVAELoss(nn.Module):
    def __init__(self):
        super(ChamferVAELoss, self).__init__()
        self.chamfer = ChamferLoss()

    def forward(self, output: torch.Tensor, mu: torch.Tensor,
                log_var: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output: output of the network
            mu: mean computed in the network
            log_var: log variance computed in the network
            real: expected value to compare to the output
        Returns:
            The variational loss computed as the sum of the hungarian loss and the Kullback-Leiber divergence.
        """
        dkl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        ch = chamfer_loss(output, real)
        return 0.0000 * torch.mean(dkl) + ch


# From dspn github
def outer(a, b=None):
    """ Compute outer product between a and b (or a and a if b is not specified). """
    if b is None:
        b = a
    size_a = tuple(a.size()) + (b.size()[-1],)
    size_b = tuple(b.size()) + (a.size()[-1],)
    a = a.unsqueeze(dim=-1).expand(*size_a)
    b = b.unsqueeze(dim=-2).expand(*size_b)
    return a, b


def chamfer_loss(predictions, targets):
    predictions = predictions.unsqueeze(0).transpose(2,3)
    targets = targets.unsqueeze(0).transpose(2,3)
    predictions, targets = outer(predictions, targets)
    squared_error = F.smooth_l1_loss(predictions, targets.expand_as(predictions), reduction="none").mean(2)
    loss = squared_error.min(2)[0] + squared_error.min(3)[0]
    return loss.view(loss.size(0), -1).mean(1)
