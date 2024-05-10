import torch


def random_in_interval(dims, min=-1, max=1):
    return (max-min) * torch.rand(dims) + min