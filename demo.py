from runObjectness import run_objectness
import cv2
from defaultParams import default_params
from drawBoxes import draw_boxes
from computeObjectnessHeatMap import compute_objectness_heat_map
import time


img_example = cv2.imread('002053.jpg')[:, :, ::-1]
params = default_params('.')
# params.cues = ['SS']

tic = time.time()
boxes = run_objectness(img_example, 10, params)
toc = time.time()

print("%f" % (toc - tic))
draw_boxes(img_example, boxes, base_color=(1, 0, 0))
compute_objectness_heat_map(img_example, boxes)
