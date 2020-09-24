import numpy as np
import mat4py
from easydict import EasyDict as edict
import os


def integrate_bayes(cues, score, params):
    likelihood = []
    for i, cue in enumerate(cues):
        if cue == 'MS':
            struct = mat4py.loadmat(os.path.join(params.data, 'MSlikelihood.mat'))
            likelihood.append(np.array(struct['likelihood']))
        elif cue == 'CC':
            struct = mat4py.loadmat(os.path.join(params.data, 'CClikelihood.mat'))
            likelihood.append(np.array(struct['likelihood']))
        elif cue == 'ED':
            struct = mat4py.loadmat(os.path.join(params.data, 'EDlikelihood.mat'))
            likelihood.append(np.array(struct['likelihood']))
        elif cue == 'SS':
            struct = mat4py.loadmat(os.path.join(params.data, 'SSlikelihood.mat'))
            likelihood.append(np.array(struct['likelihood']))
        else:
            raise Exception('Unknown cue')

    bin_number = []
    for cue_id, cue in enumerate(cues):
        if cue == 'MS':
            bin_number.append(np.maximum(np.minimum(np.ceil(score[:, cue_id] + 0.5), params.MS.numberBins + 1), 1))
        elif cue == 'CC':
            bin_number.append(np.maximum(np.minimum(np.ceil(score[:, cue_id] * 100 + 0.5), params.CC.numberBins + 1), 1))
        elif cue == 'ED':
            bin_number.append(np.maximum(np.minimum(np.ceil(score[:, cue_id] * 2 + 0.5), params.ED.numberBins + 1), 1))
        elif cue == 'SS':
            bin_number.append(np.maximum(np.minimum(np.ceil(score[:, cue_id] * 100 + 0.5), params.SS.numberBins + 1), 1))
        else:
            raise Exception('Unknown cue')

    p_obj = params.pobj
    score_bayes = np.zeros(len(score))
    bin_number = np.array(bin_number, dtype=np.int)
    bin_number -= 1     # From matlab index to numpy index

    for bb_id in range(len(score_bayes)):
        temp_pos = 1
        temp_neg = 1

        for cue_id in range(len(cues)):
            temp_pos *= likelihood[cue_id][0, bin_number[cue_id][bb_id]]
            temp_neg *= likelihood[cue_id][1, bin_number[cue_id][bb_id]]

        denominator = (temp_pos * p_obj + temp_neg * (1 - p_obj))
        if denominator:
            score_bayes[bb_id] = temp_pos * p_obj / (temp_pos * p_obj + temp_neg * (1 - p_obj))

    score_bayes += np.finfo(float).eps
    return score_bayes
