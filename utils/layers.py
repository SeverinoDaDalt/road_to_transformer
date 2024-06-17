from utils.utils import random_in_interval, batchify
import torch


class Layer:

    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_output, learning_rate=0):
        raise NotImplementedError


class Linear(Layer):

    def __init__(self, input_size, output_size, bias=True):
        self.weights = random_in_interval((input_size + int(bias), output_size))
        self.state = None

    def forward(self, input):
        batch_size = input.size(0)
        self.state = input
        return torch.einsum("bx,xy->by", batchify(input, batch_size), self.weights)

    def backward(self, gradient_output, learning_rate=0):
        if self.state is None:
            raise Exception("[layers.py] No state saved. Probably caused by calling .backprop without "
                            "previously calling .feedforward.")
        batch_size = gradient_output.size(0)
        next_gradient_output = torch.einsum("bx,xy->by", gradient_output, self.weights.T)
        gradients = torch.einsum("bx,by->xy", batchify(self.state, batch_size), gradient_output)
        self.weights -= learning_rate * gradients
        self.state = None
        return next_gradient_output[:,1:]  # We do not care about bias gradients


class Sigmoid(Layer):

    def __init__(self):
        self.output_state = None

    def forward(self, input):
        self.output_state = 1 / (1 + torch.exp(-input))
        return self.output_state

    def backward(self, gradient_output, learning_rate=0):
        if self.output_state is None:
            raise Exception("[layers.py] No state saved. Probably caused by calling .backprop without "
                            "previously calling .feedforward.")
        der = self.output_state * (1 - self.output_state)
        return torch.einsum("by,by->by", der, gradient_output)
