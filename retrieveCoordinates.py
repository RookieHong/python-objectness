import numpy as np
import math


def retrieve_coordinates(index, scale):
    img_area = scale * scale
    index1 = index % img_area
    index2 = np.floor(index / img_area)

    x1 = index1 % scale
    y1 = np.floor(index1 / scale)

    x2 = index2 % scale
    y2 = np.floor(index2 / scale)

    return np.vstack([x1, y1, x2, y2]).transpose((1, 0))
