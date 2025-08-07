import numpy as np

"""Helper functions for initializing weights"""
# for ReLU
def he(shape):
    return np.random.randn(*shape) * np.sqrt(2 / shape[0])


# for other
def xavier(shape):
    return np.random.randn(*shape) * np.sqrt(1 / shape[0])