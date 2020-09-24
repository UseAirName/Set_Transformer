import torch
import torch.nn as nn


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, set1, set2):
        """
        Args:
            set1: Tensor [batch size, point per set, features per point]
            set2: Tensor [batch size, point per set, features per point]
            The number of features must be the same for both sets
        """
        bs1, n, d1 = set1.size()
        bs2, m, d2 = set2.size()
        assert(d1 == d2 and bs1 == bs2)

        matrix1 = set1.repeat(m, 1, 1)
        matrix2 = set2.repeat(n, 1, 1)

        matrix1 = matrix1.view(bs1, n, m, -1)
        matrix2 = matrix2.view(bs1, m, n, -1).transpose(1, 2)

        diff = torch.add(matrix1, torch.neg(matrix2))
        dist = torch.norm(diff, 2, dim=3)
        out_dist, _ = torch.min(dist, dim=2)

        total_dist = torch.sum(out_dist)

        return total_dist

