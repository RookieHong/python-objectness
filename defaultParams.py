from easydict import EasyDict as edict
import os
import numpy as np


def default_params(root_dir):
    params = edict()

    # params in general
    params.min_window_height = 10
    params.min_window_width = 10
    params.distribution_windows = 100000
    params.sampled_windows = 1000
    params.trainingImages = os.path.join(root_dir, 'Training/Images')
    params.trainingExamples = os.path.join(root_dir, 'Training/Images/Examples')
    params.imageType = 'jpg'
    params.data = os.path.join(root_dir, 'Data')
    params.yourData = os.path.join(root_dir, 'Data/yourData')
    params.pobj = 0.0797
    params.tempdir = os.path.join(root_dir, 'tmpdir')
    params.pascalThreshold = 0.5
    params.cues = ['MS', 'CC', 'SS']  # full objectness measure
    params.sampling = 'nms'  # alternative sampling method - 'multinomial'

    # params for MS
    params.MS = edict()
    params.MS.name = 'Multiscale-Saliency'
    params.MS.colortype = 'rgb'
    params.MS.filtersize = 3
    params.MS.scale = np.array([16, 24, 32, 48, 64])
    params.MS.theta = np.array([0.43, 0.32, 0.34, 0.35, 0.26])
    params.MS.domain = np.tile(np.arange(0.01, 1.01, 0.01), (5, 1))
    params.MS.sizeNeighborhood = 7
    params.MS.bincenters = np.arange(0, 501, 1)
    params.MS.numberBins = len(params.MS.bincenters) - 1

    # params for CC
    params.CC = edict()
    params.CC.name = 'Color-Contrast'
    params.CC.theta = 100
    params.CC.domain = np.arange(1, 201, 1)
    params.CC.quant = np.array([4, 8, 8])
    params.CC.bincenters = np.arange(0, 2.01, 0.01)
    params.CC.numberBins = len(params.CC.bincenters) - 1

    # params for ED
    params.ED = edict()
    params.ED.name = 'Edge-Density'
    params.ED.theta = 17
    params.ED.domain = np.arange(1, 101, 1)
    params.ED.crop_size = 200
    params.ED.pixelDistance = 8
    params.ED.imageBorder = 0
    params.ED.bincenters = np.arange(0, 5.05, 0.05)
    params.ED.numberBins = len(params.ED.bincenters) - 1

    # params for SS
    params.SS = edict()
    params.SS.name = 'Superpixels-Straddling'
    params.SS.basis_sigma = 0.5
    params.SS.theta = 450
    params.SS.domain = np.arange(200, 2025, 25)
    params.SS.basis_min_area = 200
    params.SS.soft_dir = os.path.join(root_dir, 'segment')
    params.SS.pixelDistance = 8
    params.SS.imageBorder = 0.05
    params.SS.bincenters = np.arange(0, 1.01, 0.01)
    params.SS.numberBins = len(params.SS.bincenters) - 1

    return params
