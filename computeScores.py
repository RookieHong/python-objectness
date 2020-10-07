import cv2
import numpy as np
from rgb2lab import rgb2lab
from computeQuantMatrix import compute_quant_matrix
from mex_functions.computeIntegralHistogram import compute_integral_histogram
from mex_functions.computeScoreContrast import compute_score_contrast
from matlab_functions.fspecial import fspecial
from matlab_functions.mat2gray import mat2gray
from mex_functions.slidingWindowComputeScore import sliding_window_compute_score
from retrieveCoordinates import retrieve_coordinates
from generateWindows import generate_windows
from computeIntegralImageScores import compute_integral_img_scores
from py_segment.py_segment import py_segment_img
from segmentArea import segment_area
from integralHistSuperpixels import integral_hist_superpixels
from c_segment.c_segment import c_segment_img
from windowComputeScores import window_compute_score


def compute_scores(img, cue, params, windows=None):
    """

    :param img:
    :param cue:
    :param params:
    :param windows: (x1, y1, x2, y2) - python indices: from 0 to (width - 1), or (height - 1).
    :return:
    """
    boxes = None
    if windows is None:
        if cue == 'MS':
            boxes = []
            height, width, _ = img.shape

            for sid in range(len(params.MS.scale)):
                scale = params.MS.scale[sid]
                threshold = params.MS.theta[sid]
                min_width = max(2, round(params.min_window_width * scale / width))
                min_height = max(2, round(params.min_window_height * scale / height))

                # number of samples per channel to be generated
                samples = round(params.distribution_windows / (len(params.MS.scale) * 3))

                for channel in range(3):
                    saliency_map = saliency_map_channel(img, channel, params.MS.filtersize, scale)  # compute the saliency map

                    thrmap = saliency_map >= threshold
                    salmap = saliency_map * thrmap
                    thrmap_integral_image = cv2.integral(thrmap.astype(np.float))
                    salmap_integral_image = cv2.integral(salmap)

                    # compute all the windows score
                    # TODO: score_scale.T?
                    score_scale = sliding_window_compute_score(saliency_map, scale, min_width, min_height, threshold,
                                                               salmap_integral_image, thrmap_integral_image)
                    # score_scale = score_scale.T

                    idx_pos = np.where(score_scale > 0)[0]
                    score_scale = score_scale[idx_pos]

                    idx_samples = np.random.choice(np.arange(len(score_scale)), samples)
                    score_scale = score_scale[idx_samples]

                    box_coordinates = retrieve_coordinates(idx_pos[idx_samples], scale)
                    box_coordinates[:, 0] *= width / scale
                    box_coordinates[:, 2] *= width / scale
                    box_coordinates[:, 1] *= height / scale
                    box_coordinates[:, 3] *= height / scale

                    boxes.append(np.hstack([box_coordinates, score_scale.reshape(-1, 1)]))

            boxes = np.array(boxes).reshape(-1, 5)
            boxes = boxes[:params.distribution_windows]             # might be more than 100,000

        elif cue == 'CC':
            windows = generate_windows(img, 'uniform', params)      # generate windows
            boxes = compute_scores(img, cue, params, windows)

        elif cue == 'ED':
            windows = generate_windows(img, 'dense', params, cue)   # generate windows
            boxes = compute_scores(img, cue, params, windows)

        elif cue == 'SS':
            windows = generate_windows(img, 'dense', params, cue)   # generate windows
            boxes = compute_scores(img, cue, params, windows)

        return boxes

    windows = windows.copy()
    windows += 1  # From python indices to matlab indices
    if cue == 'MS':
        height, width, _ = img.shape

        ms_scores = np.zeros((len(windows), len(params.MS.scale)))   # (num_windows, scales)
        for sid in range(len(params.MS.scale)):
            scale = params.MS.scale[sid]
            threshold = params.MS.theta[sid]

            cur_scale_windows = (windows - 1) * scale / [width, height, width, height]  # This must be done with python indices

            channel_scores = np.zeros((len(windows), 3))
            for channel in range(3):
                saliency_map = saliency_map_channel(img, channel, params.MS.filtersize, scale)  # compute the saliency map

                thrmap = saliency_map >= threshold
                salmap = saliency_map * thrmap
                thrmap_integral_image = cv2.integral(thrmap.astype(np.float))
                salmap_integral_image = cv2.integral(salmap)

                channel_scores[:, channel] = window_compute_score(cur_scale_windows, scale, salmap_integral_image, thrmap_integral_image)
            ms_scores[:, sid] = np.sum(channel_scores, axis=1)
        boxes = np.hstack([windows + 1, np.sum(ms_scores, axis=1, keepdims=True)])  # matlab indices

    elif cue == 'CC':
        height, width, _ = img.shape

        lab_img = rgb2lab(img)
        Q = compute_quant_matrix(lab_img, params.CC.quant)
        integral_histogram = compute_integral_histogram(Q, height, width, np.prod(params.CC.quant))     # (prod_quant * (H + 1) * (W + 1)), col-first.

        xmin = np.round(windows[:, 0])
        ymin = np.round(windows[:, 1])
        xmax = np.round(windows[:, 2])
        ymax = np.round(windows[:, 3])

        # compute the CC score for the windows
        score = compute_score_contrast(integral_histogram, height, width, xmin, ymin, xmax, ymax, params.CC.theta,
                                       np.prod(params.CC.quant), len(windows))
        boxes = np.hstack([windows, score.reshape(-1, 1)])

    elif cue == 'ED':
        if img.shape[2] == 3:   # compute the canny map for 3 channel images
            edge_map = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 50, 150)    # TODO: Auto thresholding
        else:
            edge_map = cv2.Canny(img, 50, 150)

        h = cv2.integral(edge_map)

        xmin = np.round(windows[:, 0]).reshape(-1, 1)
        ymin = np.round(windows[:, 1]).reshape(-1, 1)
        xmax = np.round(windows[:, 2]).reshape(-1, 1)
        ymax = np.round(windows[:, 3]).reshape(-1, 1)

        xmax_inner = np.round((xmax * (200 + params.ED.theta) / (params.ED.theta + 100) + xmin * params.ED.theta / (
                    params.ED.theta + 100) + 100 / (params.ED.theta + 100) - 1) / 2)
        xmin_inner = np.round(xmax + xmin - xmax_inner)
        ymax_inner = np.round((ymax * (200 + params.ED.theta) / (params.ED.theta + 100) + ymin * params.ED.theta / (
                     params.ED.theta + 100) + 100 / (params.ED.theta + 100) - 1) / 2)
        ymin_inner = np.round(ymax + ymin - ymax_inner)

        score_windows = compute_integral_img_scores(h, windows[:, :4]).reshape(-1, 1)
        score_inner_windows = compute_integral_img_scores(h, np.hstack([xmin_inner, ymin_inner, xmax_inner, ymax_inner])).reshape(-1, 1)
        area_windows = (xmax - xmin + 1) * (ymax - ymin + 1)
        area_inner_windows = (xmax_inner - xmin_inner + 1) * (ymax_inner - ymin_inner + 1)
        area_diff = area_windows - area_inner_windows
        area_diff[area_diff == 0] = np.inf

        score = ((xmax - xmax_inner + ymax - ymax_inner) / 2) * (score_windows - score_inner_windows) / area_diff
        boxes = np.hstack([windows, score.reshape(-1, 1)])

    elif cue == 'SS':
        basis_sigma = params.SS.basis_sigma
        basis_k = params.SS.theta
        basis_min_area = params.SS.basis_min_area

        I = img
        I_area = I.shape[0] * I.shape[1]
        sf = np.sqrt(I_area / (300 * 200))
        sigma = basis_sigma * sf
        min_area = basis_min_area * sf
        k = basis_k

        S = c_segment_img(img=img, sigma=sigma, k=k, min_area=min_area)                         # C segment
        # S = py_segment_img(img=img, sigma=sigma, neighbor=4, K=k, min_comp_size=min_area)     # Python segment

        _, _, S = np.unique(S, return_index=True, return_inverse=True)
        S = S.reshape(I.shape[0], I.shape[1])
        superpixels = segment_area(S)

        integral_hist = integral_hist_superpixels(S)

        xmin = np.round(windows[:, 0])
        ymin = np.round(windows[:, 1])
        xmax = np.round(windows[:, 2])
        ymax = np.round(windows[:, 3])

        area_superpixels = np.array(superpixels['area'])
        area_windows = (xmax - xmin + 1) * (ymax - ymin + 1)

        intersection_superpixels = np.zeros((len(xmin), integral_hist.shape[2]))

        for dim in range(integral_hist.shape[2]):
            intersection_superpixels[:, dim] = compute_integral_img_scores(integral_hist[:, :, dim], windows)

        score = np.ones(len(windows)) - (np.sum(np.minimum(intersection_superpixels, np.tile(area_superpixels, (len(windows), 1)) - intersection_superpixels), axis=1) / area_windows)
        boxes = np.hstack([windows, score.reshape(-1, 1)])

    else:
        print("Option not known: check the cue names")

    boxes[:, :4] -= 1   # From matlab indices to python indices

    return boxes


def saliency_map_channel(img, channel, filter_size, scale):
    img = img[:, :, channel].astype(np.float)
    img = cv2.resize(img, (scale, scale), interpolation=cv2.INTER_LINEAR)

    # Spectral Residual
    myFFT = np.fft.fft2(img)
    myLogAmplitude = np.log(np.abs(myFFT))
    myPhase = np.angle(myFFT)
    # filter = np.ones((filter_size, filter_size), dtype=np.float) / (filter_size * filter_size)
    mySmooth = cv2.filter2D(myLogAmplitude, -1, fspecial('average', filter_size), borderType=cv2.BORDER_REPLICATE)
    mySpectralResidual = myLogAmplitude - mySmooth
    saliency_map = np.abs(np.fft.ifft2(np.exp(mySpectralResidual + 1j * myPhase))) ** 2

    # After Effect
    saliency_map = cv2.filter2D(saliency_map, -1, fspecial('disk', filter_size))
    saliency_map = mat2gray(saliency_map)
    return saliency_map
