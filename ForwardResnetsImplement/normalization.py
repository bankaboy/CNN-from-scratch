import numpy as np

class BatchNormalization:
    def __init__(self, gamma, beta, eps=1e-5):
        self.gamma = gamma
        self.beta = beta
        self.eps = eps

    def forwardBatchNormalization(self, data):
        sample_mean = data.mean(axis=0)
        sample_var = data.var(axis=0)
        
        data_norm = (data - sample_mean) / np.sqrt(sample_var + self.eps)
        data_out = self.gamma * data_norm + self.beta

        self.output = data_out