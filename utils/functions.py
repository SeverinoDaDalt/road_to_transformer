import torch


class Sigmoid:
    def f(self, x):
        return 1 / (1 + torch.exp(-x))

    def der(self, x):
        return self.f(x) * (1 - self.f(x))


class Identity:
    def f(self, x):
        return x

    def der(self, x):
        return torch.ones(x.shape)


class MeanSquaredError:

    def __init__(self, correction=0):
        self.correction = correction

    def f(self, x, y):
        """
        x and y are 2D tensors. First dimension is expected to be the batch dimension.
        """
        n = x.shape[1]
        non_normalized = ((x - y) ** 2)
        normalized = torch.einsum("bh -> b", non_normalized) / n
        return normalized + self.correction

    def der(self, x, y):
        n = x.shape[1]
        return (x - y) * 2 / n
