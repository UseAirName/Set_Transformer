from math import acos
import numpy as np
import torch


class Set:
    def __init__(self, size: int, dimension: int):
        self.size = size
        self.dimension = dimension
        self.set = np.array([])

    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.set)


class RecTriangle(Set):
    # A class for a rectangle triangle set
    def __init__(self):
        super(RecTriangle, self).__init__(3, 2)
        fst_point = np.random.random(2)
        vec1 = np.random.random(2)
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = np.array([-vec1[1], vec1[0]])
        snd_point = fst_point + np.random.rand() * vec1
        thd_point = fst_point + np.random.rand() * vec2
        point_list = [fst_point, snd_point, thd_point]
        np.random.shuffle(point_list)
        self.set = np.array(point_list)


class CovMatrixSet(Set):
    # A class for a set of point generated according a given covariance matrix
    def __init__(self, size: int, dimension: int, cov_matrix: np.ndarray):
        super(CovMatrixSet, self).__init__(size, dimension)
        x = np.random.standard_normal((size, dimension))
        self.set = x.dot(np.linalg.cholesky(cov_matrix))


def triangle_score(x: torch.Tensor):
    x = x.detach().numpy()
    scores = []
    for x_np in x:
        a1, a2, a3 = x_np[1]-x_np[0], x_np[2]-x_np[0], x_np[2]-x_np[1]
        a1, a2, a3 = a1/np.linalg.norm(a1), a2/np.linalg.norm(a2), a3/np.linalg.norm(a3)
        scores.append(max(acos(max(-1, min(1, a1.dot(a2)))),
                          acos(max(-1, min(1, a2.dot(a3)))),
                          acos(max(-1, min(1, -a1.dot(a3)))))/np.pi*180)
    return scores


def matrix_score(x: torch.Tensor, cov_matrix):
    x = x.detach().numpy()
    scores = []
    for x_np in x:
        x_np = x_np.reshape(-1)
        scores.append(np.sum(((np.cov(x_np) - cov_matrix)**2).sum(axis=1)))
    return scores
