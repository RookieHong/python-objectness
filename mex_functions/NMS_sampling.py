import numpy as np
from numba import jit


@jit(nopython=True)
def NMS_sampling(area, overlap, xmin, ymin, xmax, ymax, num_windows):
    ndx = np.ones(num_windows, dtype=np.int32) * -1
    total = len(xmin)
    visited = np.zeros(total, dtype=np.int32)

    ndx_not_visited = 0
    for w in range(num_windows):
        ndx[w] = ndx_not_visited
        visited[ndx_not_visited] = 1

        for j in range(ndx_not_visited + 1, total):
            xx1 = max(xmin[ndx_not_visited], xmin[j])
            yy1 = max(ymin[ndx_not_visited], ymin[j])
            xx2 = min(xmax[ndx_not_visited], xmax[j])
            yy2 = min(ymax[ndx_not_visited], ymax[j])

            width = xx2 - xx1 + 1
            height = yy2 - yy1 + 1

            if width > 0 and height > 0:
                ov = (width * height) / (area[ndx_not_visited] + area[j] - width * height)
                if ov > 0.5:
                    visited[j] = 1

        while ndx_not_visited < total and visited[ndx_not_visited] > 0:
            ndx_not_visited += 1

        if ndx_not_visited == total:
            break

    return ndx, visited
