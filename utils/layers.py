from utils.utils import random_in_interval, batchify
import torch


class Layer:

    def forward(self, input_):
        raise NotImplementedError

    def backward(self, grad_output, learning_rate=0):
        raise NotImplementedError


class Linear(Layer):

    def __init__(self, input_size, output_size, bias=True):
        self.weights = random_in_interval((input_size + int(bias), output_size))
        self.input_state = None

    def forward(self, input_):
        batch_size = input_.size(0)
        self.input_state = input_
        return torch.einsum("bx,xy->by", batchify(input_, batch_size), self.weights)

    def backward(self, gradient_output, learning_rate=0):
        if self.input_state is None:
            raise Exception("[layers.py] No state saved. Probably caused by calling .backprop without "
                            "previously calling .feedforward.")
        batch_size = gradient_output.size(0)
        next_gradient_output = torch.einsum("bx,xy->by", gradient_output, self.weights.T)
        gradients = torch.einsum("bx,by->xy", batchify(self.input_state, batch_size), gradient_output)
        self.weights -= learning_rate * gradients
        self.input_state = None
        return next_gradient_output[:,1:]  # We do not care about bias gradients


class Sigmoid(Layer):

    def __init__(self):
        self.output_state = None

    def forward(self, input_):
        self.output_state = 1 / (1 + torch.exp(-input_))
        return self.output_state

    def backward(self, gradient_output, learning_rate=0):
        if self.output_state is None:
            raise Exception("[layers.py] No state saved. Probably caused by calling .backprop without "
                            "previously calling .feedforward.")
        der = self.output_state * (1 - self.output_state)
        return torch.einsum("by,by->by", der, gradient_output)


class Softmax(Layer):

    def __init__(self):
        self.input_state = None
        self.output_state = None

    def forward(self, input_):
        self.input_state = input_  # bh
        expo = torch.exp(input_)  # bh
        sum_ = torch.einsum("bx->b", expo)  # b
        self.output_state = torch.einsum("bx,b->bx", expo, (1 / sum_))  # bh
        return self.output_state

    def backward(self, gradient_output, learning_rate=0):
        if self.input_state is None or self.output_state is None:
            raise Exception("[layers.py] No state saved. Probably caused by calling .backprop without "
                            "previously calling .feedforward.")
        batch_size = gradient_output.size(0)
        hidden_size = gradient_output.size(1)
        expo = torch.exp(self.input_state)  # bh
        sum_ = torch.einsum("bx->b", expo)  # b
        # computation of coef := (Id * T - [e^x, ..., e^x]) / T
        aux1 = (torch.eye(hidden_size) * sum_).repeat(batch_size, 1, 1)  # bhh
        aux2 = torch.exp(self.input_state).unsqueeze(2).repeat(1, 1, hidden_size)  # bhh
        coef = torch.einsum("bxy,b->bxy", (aux1 - aux2), (1/sum_))  # bhh
        return torch.einsum("bxy,by->bx", coef, gradient_output)
