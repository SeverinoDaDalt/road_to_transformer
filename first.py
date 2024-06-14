import torch
from utils.functions import Sigmoid, Identity, MeanSquaredError
from utils.utils import random_in_interval
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
        res = x
        for i in range(self.n_layers):
            if self.debug: print(f"Layer {i}:")
            res = torch.cat((torch.tensor([1.]), res))  # add bias in position 0
            res = res @ self.layers[i]
            self.product_states[i] = res
            if self.debug: print(f" - matrix mult:{res}")
            res = self.activations[i].f(res)
            self.output_states[i] = res
            if self.debug: print(f" - activation: {res}")
        self.output_states[-1] = x
        return res

    def backprop(self, y):
        first = True
        for i in range(self.n_layers-1, -1, -1):
            if self.product_states[i] is None or self.output_states[i] is None or self.output_states[i-1] is None:
                raise Exception("[BACKPROPAGATION] No states saved. Probably caused by calling .backprop without "
                                "previously calling .feedforward.")
            if first:
                deltas = self.activations[i].der(self.product_states[i]) * self.loss.der(self.output_states[i], y)
                first = False
            else:  # here we can use deltas from previous step
                # we skip bias (first row) for delta computation, since it has no implication in previous layers
                deltas = self.activations[i].der(self.product_states[i]) * (deltas @ self.layers[i+1][1:,:].T)
            # Following line uses:
            #  - outer product: ger(X,Y) = [xxx]^T · [yyy] -> sizeX x sizeY matrix
            #  - adding output_state for bias = 1 in position 0
            gradients = torch.ger(torch.cat((torch.tensor([1.]), self.output_states[i-1])), deltas)
            self.layers[i] -= self.lr * gradients
        # empty gradients
        self.product_states = [None] * self.n_layers
        self.output_states = [None] * (self.n_layers + 1)  # +1 because at the end I store the input


def main():
    # TODO:
    #  - allow batches

    input_size = 6
    output_size = 1
    hidden_dimensions = [input_size, 4, 4, output_size]
    mid_activation = Sigmoid()
    final_activation = Identity()
    loss = MeanSquaredError()
    lr = 1e-3
    n_train = 1000_000
    n_valid = 200
    n_test = 20

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
    train = dataset.train_iterator()

    # train
    print("[first.py] Training.")
    for i, (sequence, y) in enumerate(train):
        nn.feedforward(torch.tensor(sequence))
        nn.backprop(torch.tensor(y))
        if i % 1_000 == 0:
            # valid
            valid = dataset.valid_iterator()
            total_loss = 0
            correct = 0
            for valid_sequence, valid_y in valid:
                output = nn.feedforward(torch.tensor(valid_sequence))
                total_loss += loss.f(torch.tensor(output), torch.tensor(valid_y))
                correct += int(abs(output - valid_y) < 0.5)
            total_loss /= n_valid
            print(f"\t[first.py] Validation {i//1000} \t-> loss: {total_loss}\taccuracy: {correct}/{n_valid}")

    # test
    test = dataset.test_iterator()
    print("[first.py] Testing.")
    for sequence, y in test:
        output = nn.feedforward(torch.tensor(sequence))
        print(f"[first.py] Sequence: {sequence}\tPrediction: {output}\tReal: {y}\tDifference: {abs(output - y)}")


if __name__ == '__main__':
    main()
