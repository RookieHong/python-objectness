import numpy as np
from computeScores import compute_scores
from integrateBayes import integrate_bayes
from nms_pascal import nms_pascal
from mex_functions.scoreSampling import score_sampling


def run_objectness(img, num_samples, params):

    if len(params.cues) == 1:   # Single cue
        distribution_boxes = compute_scores(img, params.cues[0], params)
        if params.sampling == 'nms':    # NMS sampling
            if len(distribution_boxes) > params.distribution_windows:
                index_samples = score_sampling(distribution_boxes[:, 4], params.distribution_windows, 1)
                distribution_boxes = distribution_boxes[index_samples]

            boxes = nms_pascal(distribution_boxes, 0.5, num_samples)

        elif params.sampling == 'multinomial':  # Multinomial sampling
            # Sample from the distribution of the scores
            index_samples = score_sampling(distribution_boxes[:, -1], num_samples, 1)
            boxes = distribution_boxes[index_samples]

        else:
            raise Exception('Sampling procedure unknown')

        return boxes

    else:   # Combination of cues
        assert 'MS' in params.cues, "ERROR: combinations have to include MS"
        assert len(np.unique(params.cues)) == len(params.cues), "ERROR: repeated cues in the combination"

        distribution_boxes = compute_scores(img, 'MS', params)
        # rearrange the cues such that 'MS' is the first cue
        if params.cues[0] != 'MS':
            for i, cue in enumerate(params.cues):
                if cue == 'MS':
                    params.cues[i] = params.cues[0]
            params.cues[0] = 'MS'

        score = np.zeros((len(distribution_boxes), len(params.cues)))
        score[:, 0] = distribution_boxes[:, -1]     # MS score in first column
        windows = distribution_boxes[:, :4]         # (N, 4) - (x1, y1, x2, y2)
        for idx in range(1, len(params.cues)):
            temp = compute_scores(img, params.cues[idx], params, windows)
            score[:, idx] = temp[:, -1]
        score_bayes = integrate_bayes(params.cues, score, params)

        if params.sampling == 'nms':
            distribution_boxes[:, 4] = score_bayes
            boxes = nms_pascal(distribution_boxes, 0.5, num_samples)

        elif params.sampling == 'multinomial':
            index_samples = score_sampling(score_bayes, num_samples, 1)
            boxes = np.hstack([windows[index_samples], score_bayes[index_samples]])

        else:
            raise Exception('Sampling procedure unknown')

        return boxes
