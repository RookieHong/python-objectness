import numpy as np
import cv2


def integral_hist_superpixels(N):
    total_segms = np.max(N)
    height, width = N.shape

    integral_hist = np.zeros((height + 1, width + 1, total_segms))

    for sid in range(total_segms):
        superpixel_map = (N == sid).astype(np.uint8)
        integral_hist[:, :, sid] = cv2.integral(superpixel_map)

    return integral_hist
