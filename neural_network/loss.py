import torch


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


class CrossEntropy:

    def f(self, x, y):
        """
        x and y are 2D tensors. First dimension is expected to be the batch dimension.
        """
        n = x.shape[1]
        non_normalized = - y * torch.log(x)
        normalized = torch.einsum("bh -> b", non_normalized) / n
        return normalized

    def der(self, x, y):
        n = x.shape[1]
        res = - y / x
        res[res!=res] = 0  # This makes so 0/0 -> nan -> 0 as suggested here: https://stackoverflow.com/questions/64751109/pytorch-when-divided-by-zero-set-the-result-value-with-0
        return res
