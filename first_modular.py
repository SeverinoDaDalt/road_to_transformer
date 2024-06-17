import torch
from utils.layers import Linear, Sigmoid
from utils.functions import MeanSquaredError
from utils.utils import batch_generator
from datasets.increasing import IncreasingDataset
from tqdm import tqdm


class NN:

    def __init__(self, layers, loss, lr, debug=False):
        self.layers = layers
        self.loss = loss
        self.debug = debug
        self.lr = lr

    def feedforward(self, x):
        res = x  # bi
        for layer in self.layers:
            res = layer.forward(res)
        return res

    def backprop(self, output, y):
        gradient_output = self.loss.der(output, y)
        for layer in reversed(self.layers):
            gradient_output = layer.backward(gradient_output, self.lr)


def main():
    input_size = 6
    output_size = 1
    layers = [
        Linear(input_size, 4),
        Sigmoid(),
        Linear(4, 4),
        Sigmoid(),
        Linear(4, 1),
    ]
    loss = MeanSquaredError()
    lr = 5e-5
    n_train = 10_000_000
    n_valid = 10_000
    n_test = 10_000
    batch_size = 10_000

    # prepare nn
    print("[first.py] Preparing nn.")
    nn = NN(layers, loss, lr, debug=False)

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
        output = nn.feedforward(torch.tensor(batch_x))
        nn.backprop(output, torch.tensor(batch_y))

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
