import numpy as np
import cv2


def compute_quant_matrix(lab_img, bins):
    """
    Compute the quantization matrix based on the 3-dimensional matrix lab_img
    :param lab_img: 
    :param bins: 
    :return: 
    """
    assert len(bins) == 3, 'Need 3 bins for quantization'

    L = lab_img[:, :, 0]
    a = lab_img[:, :, 1]
    b = lab_img[:, :, 2]

    ll = np.minimum(np.floor(L / (100 / bins[0])) + 1, bins[0])
    aa = np.minimum(np.floor((a + 120) / (240 / bins[1])) + 1, bins[1])
    bb = np.minimum(np.floor((b + 120) / (240 / bins[2])) + 1, bins[2])

    Q = (ll - 1) * bins[1] * bins[2] + (aa - 1) * bins[2] + bb
    return Q