import torch
import torch.nn as nn


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

        matrix1 = set1.repeat(m, 1, 1)
        matrix2 = set2.repeat(n, 1, 1)

        matrix1 = matrix1.view(bs1, n, m, -1)
        matrix2 = matrix2.view(bs1, m, n, -1).transpose(1,2)

        diff = torch.add(matrix1, torch.neg(matrix2))
        dist = torch.norm(diff, 2, dim=3)

        out_dist, _ = torch.min(dist, dim=2)
        out_dist2, _ = torch.min(dist, dim=1)

        total_dist = torch.sum(out_dist)+torch.sum(out_dist2)

        return total_dist
