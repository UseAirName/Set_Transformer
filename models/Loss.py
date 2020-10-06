import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

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

        total_dist = torch.sum(out_dist) + torch.sum(out_dist2)

        return total_dist


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

        assert(d1 == d2 and bs1 == bs2) # Both sets must have the same dimension to be compared

        batch_dist = torch.cdist(set1, set2, 2)
        numpy_batch_dist = batch_dist.detach().numpy()
        indices = map(linear_sum_assignment, numpy_batch_dist)
        loss = [dist[row_idx, col_idx].sum() for dist, (row_idx, col_idx) in zip(batch_dist, indices)]

        # Mean loss computed on the batch
        total_loss = torch.mean(torch.stack(loss))
        return total_loss


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
        dkl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return dkl + self.hungarian(output, real)

