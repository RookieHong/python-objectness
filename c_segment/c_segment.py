from .build.c_segment import segment_img
import numpy as np
import cv2


def c_segment_img(img, sigma, k, min_area):
    img = np.ascontiguousarray(img)
    height, width, _ = img.shape
    segmented_img = segment_img(img.reshape(-1), height, width, float(sigma), float(k), int(min_area))
    return np.array(segmented_img).reshape(height, width)


if __name__ == '__main__':
    img = cv2.imread('002053.jpg')[:, :, ::-1]
    segmented_img = c_segment_img(img, sigma=0.8354, k=450, min_area=334)
    pass
