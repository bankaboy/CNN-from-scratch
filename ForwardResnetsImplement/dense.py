import numpy as np

# class layerDense:
class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.random.uniform(-2,2,n_neurons)

    def forwardDense(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Flatten:
    def __init__(self, convMaps):
        self.output = convMaps.flatten()