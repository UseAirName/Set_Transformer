import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


class MNISTSet:
    def __init__(self, train=True, path="MNIST", max_size=32):
        """
        Args:
            train: indicates if the Set is used for train or test
            path: path where the set is downloaded or loaded
            max_size: size of a set
        """
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        mnist_set = torchvision.datasets.MNIST(train=train, download=True, root=path, transform=transform)
        self.max_size = max_size
        self.threshold = 0.0

        self.dataset = []
        for img, label in mnist_set:
            point_set, size = self.to_set(img)
            self.dataset.append(point_set.numpy())
        self.dataset = torch.tensor(self.dataset)

    def to_set(self, img):
        """
        Args:
            img: The image to turn into a set
        """
        # selecting coordinates of points that are above the threshold
        points = torch.nonzero((img.squeeze(0) > self.threshold)).transpose(0, 1)
        size = points.size(1)
        # Random permutations of the points
        points = points[:, torch.randperm(size)]
        # Keeping or padding
        if size > self.max_size:
            points = points[:, :self.max_size]
        elif size < self.max_size:
            padding = torch.zeros(2, self.max_size - size)
            points = torch.cat((points, padding), dim=1)
        # Normalization of the points
        return torch.true_divide(points.transpose(0, 1), img.size(1)), size
