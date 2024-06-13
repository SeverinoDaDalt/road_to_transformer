import torch


def random_in_interval(dims, min=-1, max=1):
    return (max-min) * torch.rand(dims) + min


def batch_generator(pair_iterator, batch_size):
    batch1 = []
    batch2 = []
    for element1, element2 in pair_iterator:
        batch1.append(element1)
        batch2.append(element2)
        if len(batch1) == batch_size:
            yield batch1, batch2
            batch1.clear()
            batch2.clear()
    if batch1:
        yield batch1, batch2



