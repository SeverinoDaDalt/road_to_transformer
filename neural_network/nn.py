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
