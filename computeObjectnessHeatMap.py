import numpy as np
import matplotlib.pyplot as plt


def compute_objectness_heat_map(img, boxes):
    map = np.zeros((img.shape[0], img.shape[1]))
    for box in boxes:
        x1, y1, x2, y2 = box[:4].astype(np.int)
        score = box[4]
        map[y1:y2, x1:x2] += score

    plt.figure()

    plt.subplot(2, 1, 1).set_title('Input image')
    plt.imshow(img)

    plt.subplot(2, 1, 2).set_title('Objectness heat map')
    plt.imshow(map, cmap=plt.get_cmap('jet'))

    plt.show()
