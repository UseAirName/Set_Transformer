import matplotlib.pyplot as plt
import numpy as np
import torch


class Set:
    def __init__(self, size: int, dimension: int):
        self.size = size
        self.dimension = dimension
        self.set = np.array([])

    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.set)

    def plot(self):
        plt.scatter(self.set[:, 0], self.set[:, 1], color=np.random.rand(3, ), s=12)


class RecTriangle(Set):
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
    def __init__(self, size: int, dimension: int, cov_matrix: np.ndarray):
        super(CovMatrixSet, self).__init__(size, dimension)
        x = np.random.random((size, dimension))
        x = x - np.mean(x)
        self.set = x * np.linalg.cholesky(cov_matrix)


def plot2d(s):
    plt.scatter(s[:, 0], s[:, 1], color=np.random.rand(3, ), s=12)
