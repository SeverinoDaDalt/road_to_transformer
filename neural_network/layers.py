from utils.utils import random_in_interval, biasify
import torch


class Layer:

    def forward(self, input_):
        raise NotImplementedError

    def backward(self, grad_output, learning_rate=0):
        raise NotImplementedError


class Linear(Layer):
    """
    Classical linear layer W @ x + b @ 1^t, where W is an input x output matrix and b is the bias.
    """

    def __init__(self, input_size, output_size, bias=True):
        self.weights = random_in_interval((input_size + int(bias), output_size))
        self.input_state = None

    def forward(self, input_):
        self.input_state = input_
        return torch.einsum(f"...x,xy->...y", biasify(input_), self.weights)

    def backward(self, gradient_output, learning_rate=0):
        """
        The derivative is the weight matrix itself.
        Weights are updated by x @ g^t where x is the input and g is the gradient coming from the next layer.
        """
        assert self.input_state is not None, ("[layers.py] No state saved. Probably caused by calling .backprop "
                                              "without previously calling .feedforward.")
        next_gradient_output = torch.einsum(f"...x,xy->...y", gradient_output, self.weights.T)
        gradients = torch.einsum(f"...x,...y->xy", biasify(self.input_state), gradient_output)
        # update weights
        self.weights -= learning_rate * gradients
        # reset states
        self.input_state = None
        return next_gradient_output[:,1:]  # We do not care about bias gradients


class DotLinear(Layer):
    """
    Linear transformation (of each logit) used in LayerNorm y = x * _gamma - _beta
    """

    def __init__(self, hidden_size):
        self.beta = random_in_interval(hidden_size)
        self.gamma = random_in_interval(hidden_size)
        self.input_state = None

    def forward(self, input_):
        batch_size = input_.size(0)
        self.input_state = input_
        output = torch.einsum("...x,x->...x", input_, self.gamma) + self.beta.repeat(batch_size, 1)
        return output

    def backward(self, gradient_output, learning_rate=0):
        """
        Trivially the derivative is _gamma itself.
        _gammas are updated by x * g where x is the input and g is the gradient coming from the next layer.
        _betas are updated by g itself.
        """
        assert self.input_state is not None, ("[layers.py] No state saved. Probably caused by calling .backprop "
                                              "without previously calling .feedforward.")
        der = self.gamma
        self.gamma -= learning_rate * torch.einsum("...x->x", self.input_state * gradient_output)
        self.beta -= learning_rate * torch.einsum("...x->x", gradient_output)  # * 1
        return torch.einsum("y,...y->...y", der, gradient_output)


class Sigmoid(Layer):
    """
    Sigmoid activation function s(x) = 1 / (1 + e^x).
    """

    def __init__(self):
        self.output_state = None

    def forward(self, input_):
        self.output_state = 1 / (1 + torch.exp(-input_))
        return self.output_state

    def backward(self, gradient_output, learning_rate=0):
        """
        s'(x) = s(x) * (1 - s(x))
        """
        assert self.output_state is not None, ("[layers.py] No state saved. Probably caused by calling .backprop "
                                               "without previously calling .feedforward.")
        der = self.output_state * (1 - self.output_state)
        # reset states
        self.output_state = None
        return torch.einsum("...y,...y->...y", der, gradient_output)


class ReLU(Layer):
    """
    ReLU activation function r(x) = max(0, x).
    """
    def __init__(self):
        self.output_state = None

    def forward(self, input_):
        self.output_state = torch.max(input_, torch.tensor(0))  # although torch.ReLU already exist
        return self.output_state

    def backward(self, gradient_output, learning_rate=0):
        """
        The derivative is zero if x < 0 and 1 if x > 0.
        Although the ReLU function is not differentiable in 0, we can assume its value in this point is zero.
        This is the standard procedure as explained here: https://stackoverflow.com/a/76396054.
        """
        assert self.output_state is not None, ("[layers.py] No state saved. Probably caused by calling .backprop "
                                               "without previously calling .feedforward.")
        der = self.output_state
        der[der <= 0] = 0
        der[der > 0] = 1
        # reset states
        self.output_state = None
        return torch.einsum("...y,...y->...y", der, gradient_output)


class Softmax(Layer):
    """
    Softmax activation function s(x) = e^x / sum(e^x).
    """
    def __init__(self):
        self.input_state = None
        self.output_state = None

    def forward(self, input_):
        self.input_state = input_  # Bh
        expo = torch.exp(input_)  # Bh
        sum_ = torch.einsum(f"...x->...", expo)  # B
        self.output_state = torch.einsum(f"...x,...->...x", expo, (1 / sum_))  # Bh
        return self.output_state

    def backward(self, gradient_output, learning_rate=0):
        """
        The derivative is (Id*sum(e^x) - [e^x, ..., e^x]) / sum(e^x)
        """
        assert self.input_state is not None and self.output_state is not None, \
            ("[layers.py] No state saved. Probably caused by calling .backprop "
             "without previously calling .feedforward.")
        batch_size = gradient_output.size(0)
        hidden_size = gradient_output.size(1)
        expo = torch.exp(self.input_state)  # Bh
        sum_ = torch.einsum(f"...x->...", expo)  # B
        # computation of coef := (Id * T - [e^x, ..., e^x]) / T
        aux1 = torch.einsum(f"...xy,...->...xy", torch.eye(hidden_size).repeat(batch_size, 1, 1), sum_)  # Bhh
        aux2 = torch.exp(self.input_state).unsqueeze(2).repeat(1, 1, hidden_size)  # Bhh
        coef = torch.einsum(f"...xy,...->...xy", (aux1 - aux2), (1/sum_))  # Bhh
        # reset states
        self.input_state = None
        self.output_state = None
        return torch.einsum(f"...xy,...y->...x", coef, gradient_output)


class MeanAndVarianceNormalization(Layer):
    """
    This layer normalizes the input by subtracting the mean and dividing by the standard deviation.
    i.e. n(x) = (x - mean) / (std + epsilon).
    Epsilon helps control instabilities around zero.
    This layer is used to build LayerNorm and other normalization layers.
    """

    def __init__(self, epsilon=1e-6):
        self.input_state = None
        self.mean = None
        self.deviation = None
        self.epsilon = epsilon  # helps with numerical stability

    def forward(self, input_):
        hidden_size = input_.size(1)
        self.input_state = input_  # bh
        self.mean = torch.einsum("...x->...", input_)  # b
        mean_extended = torch.einsum("...,x->...x", self.mean, torch.ones((hidden_size)))  # bh
        self.deviation = torch.sqrt(torch.einsum("...x->...", (input_ - mean_extended)**2) / hidden_size)  # b
        deviation_extended = torch.einsum("...,x->...x", self.deviation, torch.ones((hidden_size)))  # bh
        output = (input_ - mean_extended) / (deviation_extended + self.epsilon)  # bh
        return output

    def backward(self, gradient_output, learning_rate=0):
        """
        The derivative is (Id - ONES/n) / (std + epsilon) - ((x - mean)(x - mean)^t) / (n * std * (std + epsilon)^2)
        """
        assert self.input_state is not None and self.mean is not None and \
               self.deviation is not None, ("[layers.py] No state saved. Probably caused by calling .backprop without "
                                            "previously calling .feedforward.")
        batch_size = gradient_output.size(0)
        hidden_size = gradient_output.size(1)
        aux1 = (torch.eye(hidden_size) - torch.ones((hidden_size, hidden_size)) / hidden_size).repeat(batch_size, 1, 1)  # bhh
        aux1 = torch.einsum("...xy,...->...xy", aux1, 1/(self.deviation + self.epsilon))  # bhh
        mean_extended = torch.einsum("...,x->...x", self.mean, torch.ones((hidden_size)))  # bh
        aux2 = torch.einsum("...x,...y->...xy", (self.input_state - mean_extended), (self.input_state - mean_extended))  # bhh
        aux2 = torch.einsum("...xy,...->...xy", aux2, 1 / (hidden_size * self.deviation * (self.deviation + self.epsilon)**2))  # bhh
        coef = aux1 - aux2
        # reset states
        self.input_state = None
        self.mean = None
        self.deviation = None
        return torch.einsum("...xy,...y->...x", coef, gradient_output)


class MatrixMultiplication(Layer):
    """
    This layer simply multiplies the two input matrices A @ B.
    """
    def __init__(self):
        self.first_matrix = None
        self.second_matrix = None

    def forward(self, first_matrix, second_matrix):
        self.first_matrix = first_matrix  # ...xy
        self.second_matrix = second_matrix  # ...yz
        output = torch.einsum("...xy,...yz->...xz", self.first_matrix, self.second_matrix)  # ...xz
        return output

    def backward(self, gradient_output, learning_rate=0):
        """
        The derivative over the first matrix is the second one and vice-versa.
        """
        assert self.first_matrix is not None and self.second_matrix is not None, \
            ("[layers.py] No state saved. Probably caused by calling .backprop without "
             "previously calling .feedforward.")
        first_gradient = torch.einsum("...yz,...xz->...xy", self.second_matrix, gradient_output)
        second_gradient = torch.einsum("...xy,...xz->...yz", self.first_matrix, gradient_output)
        self.first_matrix = None
        self.second_matrix = None
        return first_gradient, second_gradient


class LayerNorm(Layer):
    """
    A standard layer that concatenates the MeanAndVarianceNormalization and a dot layer.
    """
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
    """
    TODO: explain
    """

    def __init__(self, x_size, z_size, hidden_attention_size, output_size, mask):
        # NOTICE: we are transposing the dimension notation used in Formal Algorithms for Transformers.

        # Layers:
        self.W_query = Linear(x_size, hidden_attention_size)
        self.W_key = Linear(z_size, hidden_attention_size)
        self.W_value = Linear(z_size, output_size)
        self.key_query_multiplication_layer = MatrixMultiplication()
        self.softmax = Softmax()
        self.score_value_multiplication_layer = MatrixMultiplication()

        # Other
        self.mask = mask
        self.hidden_attention_size = hidden_attention_size

    def forward(self, x_input, z_input=None):
        # TODO: b -> ...
        # Notation: c -> context of x, k -> context of z
        if z_input is None:
            z_input = x_input
        query = self.W_query.forward(x_input)  # bxa  Notation: b -> batch, x -> context of x, a -> attention
        key = self.W_key.forward(z_input)  # bza  Notation: z -> context of z
        value = self.W_value.forward(z_input)  # bzo Notation: o -> output
        # NOTICE: tensor.mT transposes last two dimensions
        # S <- K^T @ Q
        score = self.key_query_multiplication_layer.forward(key, query.mT)  # bzx
        score = torch.einsum("bzx,zx->bzx", score, self.mask)  # bzx
        softmaxed_score = self.softmax.forward(score / torch.sqrt(self.hidden_attention_size))  # bzx
        # output = torch.einsum("bzo,bzx->bxo", value, softmaxed_score)  # bxo
        output = self.score_value_multiplication_layer.forward(softmaxed_score.mT, value)   # bxo
        return output

    def backward(self, gradient_output, learning_rate=0):
        """
        TODO: explain
        """
        score_gradient, value_gradient = self.score_value_multiplication_layer.backward(gradient_output, learning_rate)  # bxz, bzo
        score_gradient = score_gradient.mT  # bzx
        z_value_gradient = self.W_value.backward(value_gradient)  # bkz
        score_unsoftmaxed_gradient = self.softmax.backward(score_gradient) / torch.sqrt(self.hidden_attention_size)  # bzx
        score_masked_gradient = torch.einsum("bzx,zx->bzx", score_unsoftmaxed_gradient, self.mask)  # bzx
        key_gradient, query_gradient = self.key_query_multiplication_layer.backward(score_masked_gradient)  # bza, bax
        query_gradient = query_gradient.mT  # bxa
        x_gradient = self.W_query.backward(query_gradient)  # bcx
        z_key_gradient = self.W_key.backward(key_gradient)  # bkz
        z_gradient = z_key_gradient + z_value_gradient  # bkz
        return x_gradient, z_gradient
