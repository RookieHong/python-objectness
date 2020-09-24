import numpy as np
from numba import jit
import random
import math


@jit(nopython=True)
def score_sampling(score_vector, num_samples, option):
    if option == 0 and (num_samples > len(score_vector)):
        raise Exception("num_samples <= length score_vector (sampling without replacement)")

    index = np.zeros(num_samples, dtype=np.int32)
    cumsum = np.cumsum(score_vector)
    score_vector_copy = score_vector.copy()

    for i in range(num_samples):
        r = random.random() * cumsum[-1]
        minim = 0
        maxim = len(score_vector) - 1
        interval_length = maxim - minim + 1

        while interval_length > 2:
            middle = math.floor((minim + maxim) / 2)
            if cumsum[middle] > r:
                maxim = middle
            else:
                minim = middle

            interval_length = maxim - minim + 1

        if cumsum[minim] > r:
            index[i] = minim
        else:
            index[i] = maxim

        if option == 0:
            j = math.floor(index[i])
            score_vector_copy[j] = 0
            cumsum[0] = score_vector_copy[0]
            for j in range(1, len(score_vector)):
                cumsum[j] = cumsum[j - 1] + score_vector_copy[j]

    # index += 1
    return index


if __name__ == '__main__':
    score_vector = np.random.rand(1000)
    score_sampling(score_vector, 10, 1)
