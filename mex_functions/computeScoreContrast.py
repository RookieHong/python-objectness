import numpy as np
from numba import jit


@jit(nopython=True)
def compute_score_contrast(integral_histogram, height, width, xmin, ymin, xmax, ymax, thetaCC, prod_quant, num_windows):
    """

    :param integral_histogram: Column-first major array, flatten from (prod_quant, (height + 1) * (width + 1)).
    :param height:
    :param width:
    :param xmin:
    :param ymin:
    :param xmax:
    :param ymax:
    :param thetaCC:
    :param prod_quant:
    :param num_windows:
    :return:
    """
    # assert len(integral_histogram.shape) == 2, "integral_histogram must be a real double matrix"
    # assert integral_histogram.shape[1] == (height + 1) * (width + 1), "integral_histogram must be a real double matrix"
    # integral_histogram = integral_histogram.reshape(-1)

    contrast = np.zeros(num_windows)

    inside = np.zeros(prod_quant)
    outside = np.zeros(prod_quant)

    inside1 = np.zeros(prod_quant)
    outside1 = np.zeros(prod_quant)

    for w in range(num_windows):
        obj_width = xmax[w] - xmin[w] + 1
        obj_height = ymax[w] - ymin[w] + 1
        sum_inside = 0

        assert (obj_width > 0) and (obj_height > 0), "Error xmax - xmin <= 0 or ymax - ymin <= 0"

        maxmax = int(prod_quant * (xmax[w] * (height + 1) + ymax[w]))
        minmin = int(prod_quant * ((xmin[w] - 1) * (height + 1) + ymin[w] - 1))
        maxmin = int(prod_quant * (xmax[w] * (height + 1) + ymin[w] - 1))
        minmax = int(prod_quant * ((xmin[w] - 1) * (height + 1) + ymax[w]))

        # maxmax = int(prod_quant * (ymax[w] * (width + 1) + xmax[w]))
        # minmin = int(prod_quant * ((ymin[w] - 1) * (width + 1) + xmin[w] - 1))
        # maxmin = int(prod_quant * ((ymin[w] - 1) * (width + 1) + xmax[w]))
        # minmax = int(prod_quant * ((ymax[w]) * (width + 1) + xmin[w] - 1))

        for k in range(prod_quant):
            inside[k] = integral_histogram[maxmax + k] + integral_histogram[minmin + k] - integral_histogram[maxmin + k] - integral_histogram[minmax + k]
            sum_inside += inside[k]

        for k in range(prod_quant):
            if sum_inside:
                inside1[k] = inside[k] / sum_inside

        offset_width = float(obj_width) * thetaCC / 200.
        offset_height = float(obj_height) * thetaCC / 200.

        xmin_surr = round(max(xmin[w] - offset_width, 1))
        xmax_surr = round(min(xmax[w] + offset_width, width))
        ymin_surr = round(max(ymin[w] - offset_height, 1))
        ymax_surr = round(min(ymax[w] + offset_height, height))

        maxmax = int(prod_quant * (xmax_surr * (height + 1) + ymax_surr))
        minmin = int(prod_quant * ((xmin_surr - 1) * (height + 1) + ymin_surr - 1))
        maxmin = int(prod_quant * (xmax_surr * (height + 1) + ymin_surr - 1))
        minmax = int(prod_quant * ((xmin_surr - 1) * (height + 1) + ymax_surr))

        # maxmax = int(prod_quant * (ymax_surr * (width + 1) + xmax_surr))
        # minmin = int(prod_quant * ((ymin_surr - 1) * (width + 1) + xmin_surr - 1))
        # maxmin = int(prod_quant * ((ymax_surr - 1) * (width + 1) + xmin_surr))
        # minmax = int(prod_quant * (ymax_surr * (width + 1) + xmin_surr - 1))

        sum_outside = 0
        for k in range(prod_quant):
            outside[k] = integral_histogram[maxmax + k] + integral_histogram[minmin + k] - integral_histogram[maxmin + k] - integral_histogram[minmax + k] - inside[k]
            sum_outside += outside[k]

        for k in range(prod_quant):
            if sum_outside:
                outside1[k] = outside[k] / sum_outside
                if outside1[k] + inside1[k]:
                    contrast[w] += (inside1[k] - outside1[k]) * (inside1[k] - outside1[k]) / (inside1[k] + outside1[k])
            else:
                contrast[w] = 0

    return contrast
