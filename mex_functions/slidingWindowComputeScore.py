import numpy as np
from numba import jit


@jit(nopython=True)
def sliding_window_compute_score(saliency_map, scale, min_width, min_height, threshold, salmap_int, thrmap_int):
    score_scale = np.zeros(scale ** 4)
    image_area = scale * scale

    # salmap_int = salmap_int.reshape(-1)
    # thrmap_int = thrmap_int.reshape(-1)

    for xmin in range(1, scale - min_width + 2):
        for ymin in range(1, scale - min_height + 2):
            for xmax in range(xmin + min_width - 1, scale + 1):
                for ymax in range(ymin + min_height - 1, scale + 1):
                    area = (xmax - xmin + 1) * (ymax - ymin + 1)
                    aval = salmap_int[ymax, xmax] + salmap_int[ymin - 1, xmin - 1] - salmap_int[ymin - 1, xmax] - salmap_int[ymax, xmin - 1]
                    athr = thrmap_int[ymax, xmax] + thrmap_int[ymin - 1, xmin - 1] - thrmap_int[ymin - 1, xmax] - thrmap_int[ymax, xmin - 1]
                    # aval = salmap_int[(scale + 1) * (xmax - 1 + 1) + (ymax - 1 + 1)] + salmap_int[(scale + 1) * (xmin - 1) + (ymin - 1)] - \
                    #        salmap_int[(scale + 1) * (xmax - 1 + 1) + (ymin - 1)] - salmap_int[(scale + 1) * (xmin - 1) + (ymax - 1 + 1)]
                    # athr = thrmap_int[(scale + 1) * (xmax - 1 + 1) + (ymax - 1 + 1)] + thrmap_int[(scale + 1) * (xmin - 1) + (ymin - 1)] - \
                    #        thrmap_int[(scale + 1) * (xmax - 1 + 1) + (ymin - 1)] - thrmap_int[(scale + 1) * (xmin - 1) + (ymax - 1 + 1)]
                    score_scale[image_area * ((ymax - 1) * scale + xmax - 1) + ((ymin - 1) * scale + xmin - 1)] = (aval * athr) / area

    # return score_scale.reshape((scale * scale, scale * scale))
    return score_scale