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
        return 1


class MeanSquaredError:

    def __init__(self, correction=0):
        self.correction = correction

    def f(self, x, y):
        return ((x - y) ** 2) / 2 + self.correction

    def der(self, x, y):
        return x - y
