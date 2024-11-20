import numpy as np

def mean_std(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std

def standardize(X, mean, std):
    result = (X - mean) /std

    return result