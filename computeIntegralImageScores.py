import numpy as np


def compute_integral_img_scores(integral_img, windows):
    """
    Computes the score of the windows wrt the integral_img
    :param integral_img:
    :param windows: (x1, y1, x2, y2)
    :return:
    """
    windows = np.round(windows).astype(np.int)

    width = integral_img.shape[1]
    integral_img = integral_img.reshape(-1)
    index1 = width * (windows[:, 3] + 1) + windows[:, 2]
    index2 = width * (windows[:, 1]) + windows[:, 0] - 1
    index3 = width * (windows[:, 3] + 1) + windows[:, 0] - 1
    index4 = width * (windows[:, 1]) + windows[:, 2]

    # Original matlab code:
    # windows[windows == 0] = 1
    # integral_img = integral_img.T.reshape(-1)
    # height = integral_img.shape[0]
    # index1 = height * windows[:, 2] + (windows[:, 3] + 1) - 1
    # index2 = height * (windows[:, 0] - 1) + windows[:, 1] - 1
    # index3 = height * (windows[:, 0] - 1) + (windows[:, 3] + 1) - 1
    # index4 = height * windows[:, 2] + windows[:, 1] - 1

    score = integral_img[index1] + integral_img[index2] - integral_img[index3] - integral_img[index4]

    return score
