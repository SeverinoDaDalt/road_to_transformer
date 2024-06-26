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


class BinaryCrossEntropy:

    def f(self, x, y):
        """
        x and y are 2D tensors. First dimension is expected to be the batch dimension.
        """
        n = x.shape[1]
        non_normalized = - (y * torch.log(x) + (1 - y) * torch.log(1 - x))
        normalized = torch.einsum("bh -> b", non_normalized) / n
        return normalized

    def der(self, x, y):
        n = x.shape[1]
        aux1 = y / x
        aux1[aux1!=aux1] = 0  # This makes so 0/0 -> nan -> 0 as suggested here: https://stackoverflow.com/questions/64751109/pytorch-when-divided-by-zero-set-the-result-value-with-0
        aux2 = (1 - y) / (1 - x)
        aux2[aux2!=aux2] = 0
        return - (aux1 - aux2)
