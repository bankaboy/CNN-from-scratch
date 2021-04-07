import numpy as np

# class reluActivation:
class Relu:
    def applyRelu(self, arr):
        self.reluActivatedOutput = np.maximum(arr, 0)


# class softmaxActivation:
class Softmax:
    def applySoftmax(self, arr):
        arr -= np.max(arr)
        exp_values = np.exp(arr)
        self.softmaxActivatedOutput = exp_values/np.sum(exp_values, keepdims=True)


# class sigmoidActivation:
class Sigmoid:
    def applySigmoid(self,arr):
        self.sigmoidActivatedOutput = 1/(1 + np.exp(-arr))