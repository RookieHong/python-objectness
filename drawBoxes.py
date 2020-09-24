import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw_boxes(img, boxes, base_color=(1, 0, 0), line_width=3):
    base_color = np.array(base_color)
    boxes = boxes[np.argsort(-boxes[:, 4])]     # Sort in descending order of score
    max_score = np.max(boxes[:, 4])
    ax = plt.gca()
    ax.imshow(img)
    for box in boxes:
        xmin, ymin, xmax, ymax, score = box
        color = base_color * score / max_score
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=line_width, edgecolor=color, fill=False)
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()
