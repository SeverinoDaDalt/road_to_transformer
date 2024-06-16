import torch
from utils.functions import Sigmoid, Identity, MeanSquaredError
from utils.utils import random_in_interval, batch_generator
from datasets.increasing import IncreasingDataset
from tqdm import tqdm


class NN:

    def __init__(self, layers, activations, loss, lr, debug=False):
        assert len(layers) == len(activations)
        self.layers = layers  # TODO
        self.activations = activations  # TODO
        self.loss = loss
        self.debug = debug
        self.lr = lr

        self.n_layers = len(layers)
        self.product_states = [None] * self.n_layers
        self.output_states = [None] * (self.n_layers + 1)  # +1 because at the end I store the input

    def feedforward(self, x):
        res = x  # bi
        batch_size = x.size(0)
        for i in range(self.n_layers):
            if self.debug: print(f"Layer {i}:")
            res = torch.cat((torch.tensor([[1.]] * batch_size).T, res.T)).T  # add bias in position 0  # TODO: make the adding of the bias a function itself
            # res = res @ self.layers[i]  # bi,ih -> bh
            # res = torch.einsum("x,xy->y", res, self.layers[i])
            res = torch.einsum("zx,xy->zy", res, self.layers[i])
            self.product_states[i] = res  # bh
            if self.debug: print(f" - matrix mult: {res.size()}")
            # TODO: this will no longer work for non-one2one functions (such as softmax), once batch is implemented
            res = self.activations[i].f(res)  # bh -> bh
            self.output_states[i] = res  # bh
            if self.debug: print(f" - activation: {res.size()}")
        self.output_states[-1] = x
        if self.debug: print(f"Input: {x.size()}")
        return res

    def backprop(self, y):
        batch_size = y.size(0)
        first = True
        for i in range(self.n_layers-1, -1, -1):
            if self.product_states[i] is None or self.output_states[i] is None or self.output_states[i-1] is None:
                raise Exception("[BACKPROPAGATION] No states saved. Probably caused by calling .backprop without "
                                "previously calling .feedforward.")
            if first:
                # bh*bh -> bh
                # deltas = self.activations[i].der(self.product_states[i]) * self.loss.der(self.output_states[i], y)
                deltas = torch.einsum("bx,bx->bx", self.activations[i].der(self.product_states[i]), self.loss.der(self.output_states[i], y))
                first = False
            else:  # here we can use deltas from previous step
                # we skip bias (first row) for delta computation, since it has no implication in previous layers
                # bg,gh -> bh
                # bh,bh -> bh
                # deltas = self.activations[i].der(self.product_states[i]) * (deltas @ self.layers[i+1][1:,:].T)
                partials = torch.einsum("bx,xy->by", deltas, self.layers[i+1][1:,:].T)  # TODO: try remove transpose
                deltas = torch.einsum("by,by->by", self.activations[i].der(self.product_states[i]), partials)
            # Following line uses:
            #  - outer product: ger(X,Y) = [xxx]^T · [yyy] -> sizeX x sizeY matrix
            #  - adding output_state for bias = 1 in position 0
            # o,bh -> oh
            # gradients = torch.ger(torch.cat((torch.tensor([1.]), self.output_states[i-1])), deltas)
            gradients = torch.einsum("bx,by->xy", torch.cat((torch.tensor([[1.]] * batch_size).T, self.output_states[i-1].T)).T, deltas)  # TODO: make the adding of the bias a function itself
            self.layers[i] -= self.lr * gradients
        # empty gradients
        self.product_states = [None] * self.n_layers
        self.output_states = [None] * (self.n_layers + 1)  # +1 because at the end I store the input


def main():
    input_size = 6
    output_size = 1
    hidden_dimensions = [input_size, 4, 4, output_size]
    mid_activation = Sigmoid()
    final_activation = Identity()
    loss = MeanSquaredError()
    lr = 5e-5
    n_train = 10_000_000
    n_valid = 10_000
    n_test = 10_000
    batch_size = 10_000

    # prepare nn
    print("[first.py] Preparing nn.")
    layers = []
    activations = [mid_activation] * (len(hidden_dimensions)-2) + [final_activation]
    for i in range(1, len(hidden_dimensions)):
        layers.append(random_in_interval((hidden_dimensions[i-1] + 1, hidden_dimensions[i])))  # +1 is the bias
    nn = NN(layers, activations, loss, lr, debug=False)

    # prepare dataset
    print("[first.py] Preparing dataset.")
    dataset = IncreasingDataset(input_size, n_train=n_train, n_valid=n_valid, n_test=n_test)
    train = batch_generator(dataset.train_iterator(), batch_size=batch_size)

    # train
    print("[first.py] Training.")
    for i, (batch_x, batch_y) in enumerate(train):
        # valid
        valid = batch_generator(dataset.valid_iterator(), batch_size=batch_size)
        total_loss = 0
        correct = 0
        for valid_x, valid_y in valid:
            output = nn.feedforward(torch.tensor(valid_x))
            total_loss += torch.einsum("xy->", loss.f(torch.tensor(output), torch.tensor(valid_y)))
            for j in range(output.size(0)):
                correct += int(abs(output[j] - torch.tensor(valid_y[j])) < 0.5)
        total_loss /= n_valid
        print(f"\t[first.py] Validation {i} \t-> loss: {total_loss}\taccuracy: {correct}/{n_valid}")
        # learn
        nn.feedforward(torch.tensor(batch_x))
        nn.backprop(torch.tensor(batch_y))

    # test
    test = batch_generator(dataset.test_iterator(), batch_size=batch_size)
    total_loss = 0
    total_correct = 0
    print("[first.py] Testing.")
    for test_x, test_y in test:
        output = nn.feedforward(torch.tensor(test_x))
        total_loss += torch.einsum("xy->", loss.f(torch.tensor(output), torch.tensor(test_y)))
        for j in range(output.size(0)):
            correct = int(abs(output[j] - torch.tensor(test_y[j])) < 0.5)
            total_correct += correct
            if not correct:
                print(f"\t[first.py] Not correct \t-> sample: {test_x[j]}\tprediction: {output[j]}\treal: {torch.tensor(test_y[j])}")
    print(f"\t[first.py] Test results \t-> loss: {total_loss}\taccuracy: {total_correct}/{n_test}")


if __name__ == '__main__':
    main()
