import torch


def random_in_interval(dims, min_=-1, max_=1):
    return (max_ - min_) * torch.rand(dims) + min_


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


def biasify(input_):
    """
    Adds one column of 1's respective to the bias of the linear model.
    """
    # First, we recover the number of "batch" dimensions (batch, context).
    # Required to take into account the extra dim given by the context in Attention.

    batch_dims = input_.shape[:-1]
    ones_hyperplane = torch.ones((*batch_dims, 1))
    return torch.cat((ones_hyperplane, input_), len(batch_dims))



