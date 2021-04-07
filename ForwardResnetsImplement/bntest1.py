# https://towardsdatascience.com/implementing-batch-normalization-in-python-a044b0369567
# https://medium.com/techspace-usict/normalization-techniques-in-deep-neural-networks-9121bf100d8#:~:text=As%20the%20name%20suggests%2C%20Group,group%20normalization%20becomes%20Layer%20normalization.

import numpy as np
from pprint import pprint

def batchnorm_forward(x, gamma, beta, eps=1e-5):
    # N, D = x.shape
    
    sample_mean = x.mean(axis=0)
    sample_var = x.var(axis=0)
    
    std = np.sqrt(sample_var + eps)
    x_centered = x - sample_mean
    x_norm = x_centered / std
    out = gamma * x_norm + beta
    
    cache = (x_norm, x_centered, std, gamma)

    return out, cache

data = np.random.randint(0,5, (3,28,28))
pprint(data)

data_normed, details = batchnorm_forward(data, 5, 2)
pprint(data_normed)