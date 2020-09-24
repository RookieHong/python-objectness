import numpy as np
from numba import jit
import math
import cv2


@jit(nopython=True)
def compute_integral_histogram(quant_matrix, height, width, prod_quant):
    # flatten_quant_matrix = quant_matrix.reshape(-1)
    int_hist = np.zeros(prod_quant * (height + 1) * (width + 1))
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            x1 = math.floor(quant_matrix[i - 1, j - 1])
            int_hist[prod_quant * (j * (height + 1) + i) + x1 - 1] = 1  # corresponding bin has value=1 at location (i,j)
            
            for k in range(prod_quant):
                int_hist[prod_quant * (j * (height + 1) + i) + k] += int_hist[prod_quant * (j * (height + 1) + i - 1) + k] + int_hist[prod_quant * ((j - 1) * (height + 1) + i) + k] - int_hist[prod_quant * ((j - 1) * (height + 1) + i - 1) + k]

    # return int_hist.reshape((height + 1) * (width + 1), prod_quant).T   # First row-major, then col-major
    return int_hist


if __name__ == "__main__":
    quant_matrix = np.random.rand(270, 480)
    int_hist = compute_integral_histogram(quant_matrix, 270, 480, 256)
    pass
