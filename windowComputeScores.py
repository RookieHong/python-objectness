import numpy as np


def window_compute_score(windows, scale, salmap_int, thrmap_int):
    score_scale = np.zeros(len(windows))
    image_area = scale * scale

    windows = windows.astype(np.int)
    xmin = windows[:, 0]
    ymin = windows[:, 1]
    xmax = windows[:, 2]
    ymax = windows[:, 3]

    area = (xmax - xmin + 1) * (ymax - ymin + 1)
    aval = salmap_int[ymax, xmax] + salmap_int[ymin - 1, xmin - 1] - salmap_int[ymin - 1, xmax] - salmap_int[ymax, xmin - 1]
    athr = thrmap_int[ymax, xmax] + thrmap_int[ymin - 1, xmin - 1] - thrmap_int[ymin - 1, xmax] - thrmap_int[ymax, xmin - 1]
    # score_scale[image_area * ((ymax - 1) * scale + xmax - 1) + ((ymin - 1) * scale + xmin - 1)] = (aval * athr) / area
    score_scale = (aval * athr) / area

    return score_scale