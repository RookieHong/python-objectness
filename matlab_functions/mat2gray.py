import numpy as np


def mat2gray(A):
    limits = np.array([np.min(A), np.max(A)], dtype=np.float)
    delta = limits[1] - limits[0]
    I = (A.astype(np.float) - limits[0]) / delta
    I = np.maximum(0, np.minimum(1, I))     # Make sure all values are between 0 and 1.
    return I