import numpy as np
from easydict import EasyDict as edict


def segment_area(N):
    segms = {
        'coords': [],
        'area': []
    }

    tot_segms = np.max(N)
    for sid in range(tot_segms):
        cols, rows = np.where(N == sid)
        segms['coords'].append(np.vstack([cols, rows]))
        segms['area'].append(len(cols))

    return segms
