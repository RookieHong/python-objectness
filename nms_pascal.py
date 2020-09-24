import numpy as np
from mex_functions.NMS_sampling import NMS_sampling


def nms_pascal(boxes, overlap, max_windows=1000):
    """
    Greedily select high-scoring detections and skip detections that are significantly covered by a previously selected
    detection.
    :param boxes:
    :param overlap:
    :param max_windows:
    :return:
    """
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    I = np.argsort(-s)
    pick, _ = NMS_sampling(area[I], overlap, x1[I], y1[I], x2[I], y2[I], max_windows)
    pick = pick[pick >= 0]
    top = boxes[I[pick], :]

    return top
