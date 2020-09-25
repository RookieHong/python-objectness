import numpy as np
import cv2
import random
import math
from numba import jit


def generate_windows(img, opt_gen, params, cue=None):
    """

    :param img:
    :param opt_gen:
    :param params:
    :param cue:
    :return: windows: (x1, y1, x2, y2) - python indices: from 0 to (width - 1), or (height - 1).
    """
    height, width, _ = img.shape

    if opt_gen == 'uniform':
        total_samples = params.distribution_windows
        min_width = params.min_window_width
        min_height = params.min_window_height

        xmin, ymin, xmax, ymax = generate_coords(opt_gen, height, width, total=total_samples, min_height=min_height, min_width=min_width)

        windows = np.hstack([xmin, ymin, xmax, ymax])
        windows -= 1    # From matlab index to python index

    elif opt_gen == 'dense':  # for SS or ED
        assert cue is not None, "cue is not specified"

        pixelDistance = params[cue].pixelDistance
        imageBorder = params[cue].imageBorder

        offsetHeight = math.floor(imageBorder * height)
        offsetWidth = math.floor(imageBorder * width)

        height = math.floor(height * (1 - imageBorder) / pixelDistance)
        width = math.floor(width * (1 - imageBorder) / pixelDistance)

        totalWindows = int(height * width * (height + 1) * (width + 1) / 4)

        xmin, ymin, xmax, ymax = generate_coords(opt_gen, height, width, total=totalWindows)

        xmin = xmin * pixelDistance + offsetWidth
        xmax = xmax * pixelDistance + offsetWidth
        ymin = ymin * pixelDistance + offsetHeight
        ymax = ymax * pixelDistance + offsetHeight

        windows = np.hstack([xmin, ymin, xmax, ymax])
        windows -= 1    # From matlab index to python index

    else:
        raise Exception('optionGenerate unknown')

    return windows


@jit(nopython=True)
def generate_coords(opt_gen, height, width, total, min_height=None, min_width=None):
    if opt_gen == 'uniform':
        assert min_height is not None and min_width is not None, "min_height and min_width must be specified"

        xmin = np.zeros((total, 1))
        ymin = np.zeros((total, 1))
        xmax = np.zeros((total, 1))
        ymax = np.zeros((total, 1))

        for j in range(total):
            x1 = round(random.random() * (width - 1) + 1)
            x2 = round(random.random() * (width - 1) + 1)
            while abs(x1 - x2) + 1 < min_width:
                x1 = round(random.random() * (width - 1) + 1)
                x2 = round(random.random() * (width - 1) + 1)

            y1 = round(random.random() * (height - 1) + 1)
            y2 = round(random.random() * (height - 1) + 1)
            while abs(y1 - y2) + 1 < min_height:
                y1 = round(random.random() * (height - 1) + 1)
                y2 = round(random.random() * (height - 1) + 1)

            xmin[j] = min(x1, x2)
            ymin[j] = min(y1, y2)
            xmax[j] = max(x1, x2)
            ymax[j] = max(y1, y2)

        return xmin, ymin, xmax, ymax

    elif opt_gen == 'dense':
        xmin = np.zeros((total, 1))
        xmax = np.zeros((total, 1))
        ymin = np.zeros((total, 1))
        ymax = np.zeros((total, 1))

        currentWindow = 0

        for x in range(1, width + 1):
            for y in range(1, height + 1):
                for w in range(1, width - x + 2):
                    for h in range(1, height - y + 2):
                        xmin[currentWindow] = x
                        ymin[currentWindow] = y
                        xmax[currentWindow] = x + w - 1
                        ymax[currentWindow] = y + h - 1
                        currentWindow += 1

        return xmin, ymin, xmax, ymax