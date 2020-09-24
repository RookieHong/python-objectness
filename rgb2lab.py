import numpy as np
import cv2
import matplotlib.pyplot as plt


def rgb2lab(img):
    # function lab_img = RGB2Lab(rgb_img)
    # RGB2Lab takes matrices corresponding to Red, Green, and Blue, and
    # % transforms them into CIELab.  This transform is based on ITU-R
    # % Recommendation  BT.709 using the D65 white point reference.
    # % The error in transforming RGB -> Lab -> RGB is approximately
    # % 10^-5.  RGB values can be either between 0 and 1 or between 0 and 255.
    # % By Mark Ruzon from C code by Yossi Rubner, 23 September 1997.
    # % Updated for MATLAB 5 28 January 1998.

    img = img.astype(np.float)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    if (np.max(R) > 1.0) | (np.max(G) > 1.0) | (np.max(B) > 1.0):
        R /= 255
        G /= 255
        B /= 255

    H, W = R.shape
    s = H * W

    # Set a threshold
    T = 0.008856

    RGB = np.stack([R.reshape(-1), G.reshape(-1), B.reshape(-1)])  # (3, H * W)

    # RGB to XYZ
    MAT = np.array([[0.412453, 0.357580, 0.180423],
                    [0.212671, 0.715160, 0.072169],
                    [0.019334, 0.119193, 0.950227]])
    XYZ = np.dot(MAT, RGB)

    X = XYZ[0, :] / 0.950456
    Y = XYZ[1, :]
    Z = XYZ[2, :] / 1.088754

    XT = X > T
    YT = Y > T
    ZT = Z > T

    fX = XT * X ** (1/3) + (~XT) * (7.787 * X + 16/116)

    # Compute L
    Y3 = Y ** (1/3)
    fY = YT * Y3 + (~YT) * (7.787 * Y + 16/116)
    L = YT * (116 * Y3 - 16.0) + (~YT) * (903.3 * Y)

    fZ = ZT * Z ** (1/3) + (~ZT) * (7.787 * Z + 16/116)

    # Compute a and b
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)

    L = L.reshape(H, W)
    a = a.reshape(H, W)
    b = b.reshape(H, W)

    L = np.stack([L, a, b]).transpose((1, 2, 0))
    return L


if __name__ == "__main__":
    img = cv2.imread('test_imgs/Bolt2_0116.jpg')[:, :, ::-1]
    lab_img = rgb2lab(img)
    plt.imshow(lab_img)
    plt.show()
