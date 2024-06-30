from utils.utils import random_in_interval, biasify
import torch


BATCH_LETTERS = "bcd"  # More letters may be needed in the future. Avoid using them in einsum...


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
        self.input_state = input_
        bs = BATCH_LETTERS[:len(input_.shape)-1]
        return torch.einsum(f"{bs}x,xy->{bs}y", biasify(input_), self.weights)

    def backward(self, gradient_output, learning_rate=0):
        assert self.input_state is not None, ("[layers.py] No state saved. Probably caused by calling .backprop "
                                              "without previously calling .feedforward.")
        bs = BATCH_LETTERS[:len(gradient_output.shape)-1]
        next_gradient_output = torch.einsum(f"{bs}x,xy->{bs}y", gradient_output, self.weights.T)
        gradients = torch.einsum(f"{bs}x,{bs}y->xy", biasify(self.input_state), gradient_output)
        # update weights
        self.weights -= learning_rate * gradients
        # reset states
        self.input_state = None
        return next_gradient_output[:,1:]  # We do not care about bias gradients


class DotLinear(Layer):
    """
    Linear transformation (of each logit) used in LayerNorm.
    y <- x * _gamma - _beta
    """

    def __init__(self, hidden_size):
        self.beta = random_in_interval(hidden_size)
        self.gamma = random_in_interval(hidden_size)
        self.input_state = None

    def forward(self, input_):
        batch_size = input_.size(0)
        self.input_state = input_
        output = torch.einsum("bx,x->bx", input_, self.gamma) + self.beta.repeat(batch_size, 1)
        return output

    def backward(self, gradient_output, learning_rate=0):
        assert self.input_state is not None, ("[layers.py] No state saved. Probably caused by calling .backprop "
                                              "without previously calling .feedforward.")
        der = self.gamma
        self.gamma -= learning_rate * torch.einsum("bx->x", self.input_state * gradient_output)
        self.beta -= learning_rate * torch.einsum("bx->x", gradient_output)  # * 1
        return torch.einsum("y,by->by", der, gradient_output)


class Sigmoid(Layer):

    def __init__(self):
        self.output_state = None

    def forward(self, input_):
        self.output_state = 1 / (1 + torch.exp(-input_))
        return self.output_state

    def backward(self, gradient_output, learning_rate=0):
        assert self.output_state is not None, ("[layers.py] No state saved. Probably caused by calling .backprop "
                                               "without previously calling .feedforward.")
        der = self.output_state * (1 - self.output_state)
        # reset states
        self.output_state = None
        return torch.einsum("by,by->by", der, gradient_output)


class ReLU(Layer):

    def __init__(self):
        self.output_state = None

    def forward(self, input_):
        self.output_state = torch.max(input_, torch.tensor(0))  # although torch.ReLU already exist
        return self.output_state

    def backward(self, gradient_output, learning_rate=0):
        assert self.output_state is not None, ("[layers.py] No state saved. Probably caused by calling .backprop "
                                               "without previously calling .feedforward.")
        der = self.output_state
        der[der <= 0] = 0  # on why this works: https://stackoverflow.com/a/76396054
        der[der > 0] = 1
        # reset states
        self.output_state = None
        return torch.einsum("by,by->by", der, gradient_output)


class Softmax(Layer):

    def __init__(self):
        self.input_state = None
        self.output_state = None

    def forward(self, input_):
        bs = BATCH_LETTERS[:len(input_.shape)-1]
        self.input_state = input_  # Bh
        expo = torch.exp(input_)  # Bh
        sum_ = torch.einsum(f"{bs}x->{bs}", expo)  # B
        self.output_state = torch.einsum(f"{bs}x,{bs}->{bs}x", expo, (1 / sum_))  # Bh
        return self.output_state

    def backward(self, gradient_output, learning_rate=0):
        # TODO: try to substitute BATCH LETTERS with ellipsis: https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
        assert self.input_state is not None and self.output_state is not None, \
            ("[layers.py] No state saved. Probably caused by calling .backprop "
             "without previously calling .feedforward.")
        bs = BATCH_LETTERS[:len(gradient_output.shape)-1]
        batch_size = gradient_output.size(0)
        hidden_size = gradient_output.size(1)
        expo = torch.exp(self.input_state)  # Bh
        sum_ = torch.einsum(f"{bs}x->{bs}", expo)  # B
        # computation of coef := (Id * T - [e^x, ..., e^x]) / T
        aux1 = torch.einsum(f"{bs}xy,{bs}->{bs}xy", torch.eye(hidden_size).repeat(batch_size, 1, 1), sum_)  # Bhh
        aux2 = torch.exp(self.input_state).unsqueeze(2).repeat(1, 1, hidden_size)  # Bhh
        coef = torch.einsum(f"{bs}xy,{bs}->{bs}xy", (aux1 - aux2), (1/sum_))  # Bhh
        # reset states
        self.input_state = None
        self.output_state = None
        return torch.einsum(f"{bs}xy,{bs}y->{bs}x", coef, gradient_output)


class MeanAndVarianceNormalization(Layer):
    """
    This layer normalizes the input by subtracting the mean and dividing by the standard deviation.
    Used to build LayerNorm and other normalization layers.
    """

    def __init__(self, epsilon=1e-6):
        self.input_state = None
        self.mean = None
        self.deviation = None
        self.epsilon = epsilon  # helps with numerical stability

    def forward(self, input_):
        batch_size = input_.size(0)
        hidden_size = input_.size(1)
        self.input_state = input_  # bh
        self.mean = torch.einsum("bx->b", input_)  # b
        mean_extended = torch.einsum("b,x->bx", self.mean, torch.ones((hidden_size)))  # bh
        self.deviation = torch.sqrt(torch.einsum("bx->b", (input_ - mean_extended)**2) / hidden_size)  # b
        deviation_extended = torch.einsum("b,x->bx", self.deviation, torch.ones((hidden_size)))  # bh
        output = (input_ - mean_extended) / (deviation_extended + self.epsilon)  # bh
        return output

    def backward(self, gradient_output, learning_rate=0):
        assert self.input_state is not None and self.mean is not None and \
               self.deviation is not None, ("[layers.py] No state saved. Probably caused by calling .backprop without "
                                            "previously calling .feedforward.")
        batch_size = gradient_output.size(0)
        hidden_size = gradient_output.size(1)
        aux1 = (torch.eye(hidden_size) - torch.ones((hidden_size, hidden_size)) / hidden_size).repeat(batch_size, 1, 1)  # bhh
        aux1 = torch.einsum("bxy,b->bxy", aux1, 1/(self.deviation + self.epsilon))  # bhh
        mean_extended = torch.einsum("b,x->bx", self.mean, torch.ones((hidden_size)))  # bh
        aux2 = torch.einsum("bx,by->bxy", (self.input_state - mean_extended), (self.input_state - mean_extended))  # bhh
        aux2 = torch.einsum("bxy,b->bxy", aux2, 1 / (hidden_size * self.deviation * (self.deviation + self.epsilon)**2))  # bhh
        coef = aux1 - aux2
        # reset states
        self.input_state = None
        self.mean = None
        self.deviation = None
        return torch.einsum("bxy,by->bx", coef, gradient_output)


class LayerNorm(Layer):

    def __init__(self, hidden_size, epsilon=1e-6):
        self.normalization_layer = MeanAndVarianceNormalization(epsilon)
        self.dotlinear_layer = DotLinear(hidden_size)

    def forward(self, input_):
        output = self.normalization_layer.forward(input_)
        output = self.dotlinear_layer.forward(output)
        return output

    def backward(self, gradient_output, learning_rate=0):
        new_gradient_output = self.dotlinear_layer.backward(gradient_output, learning_rate)
        new_gradient_output = self.normalization_layer.backward(new_gradient_output, learning_rate)
        return new_gradient_output


class Attention(Layer):

    def __init__(self, x_size, z_size, hidden_attention_size, output_size, mask):
        # NOTICE: we are transposing the dimension notation used in Formal Algorithms for Transformers.
        self.W_query = Linear(x_size, hidden_attention_size)
        self.W_key = Linear(z_size, hidden_attention_size)
        self.W_value = Linear(z_size, output_size)
        self.softmax = Softmax()
        self.mask = mask
        self.hidden_attention_size = hidden_attention_size

    def forward(self, x_input, z_input=None):
        if z_input is None:
            z_input = x_input
        query = self.W_query.forward(x_input)  # bxa  Notation: b -> batch, x -> context of x, a -> attention
        key = self.W_key.forward(z_input)  # bza  Notation: z -> context of z
        value = self.W_value.forward(z_input)  # bzo Notation: o -> output
        # S <- K^T @ Q
        score = torch.einsum("bza,bxa->bzx", key, query)  # bzx
        score = torch.einsum("bzx,zx->bzx", score, self.mask)  # bzx
        softmaxed_score = self.softmax.forward(score / torch.sqrt(self.hidden_attention_size))  # bzx
        output = torch.einsum("bzo,bzx->bxo", value, softmaxed_score)  # bxo
        # TODO: save state
        return output

    def backward(self, gradient_output, learning_rate=0):
        # TODO
        raise NotImplementedError
