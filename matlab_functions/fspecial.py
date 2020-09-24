import numpy as np
import math


def fspecial(type, radius):
    if type == 'average':
        return np.ones((radius, radius), dtype=np.float) / (radius * radius)
    elif type == 'disk':
        crad = math.ceil(radius - 0.5)
        x, y = np.meshgrid(np.arange(-crad, crad + 1), np.arange(-crad, crad + 1))

        maxxy = np.maximum(np.abs(x), np.abs(y))
        minxy = np.minimum(np.abs(x), np.abs(y))
        m1 = (radius ** 2 < (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * (minxy - 0.5) + \
             (radius ** 2 >= (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * np.sqrt(radius ** 2 - (maxxy + 0.5) ** 2, dtype=np.complex)
        m2 = (radius ** 2 > (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * (minxy + 0.5) + \
             (radius ** 2 <= (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * np.sqrt(radius ** 2 - (maxxy - 0.5) ** 2, dtype=np.complex)
        sgrid = (radius ** 2 * (0.5 * (np.arcsin(m2 / radius) - np.arcsin(m1 / radius)) + 0.25 *
                (np.sin(2 * np.arcsin(m2 / radius)) - np.sin(2 * np.arcsin(m1 / radius)))) -
                (maxxy - 0.5) * (m2 - m1) + (m1 - minxy + 0.5)) * \
                ((((radius ** 2 < (maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2) & (radius ** 2 > (maxxy - 0.5) ** 2 + (minxy - 0.5) ** 2)) |
                ((minxy == 0) & (maxxy - 0.5 < radius) & (maxxy + 0.5 >= radius))))

        sgrid = sgrid + ((maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2 < radius ** 2)
        sgrid[crad + 1, crad + 1] = min(np.pi * radius ** 2, np.pi / 2)
        if (crad > 0) and (radius > crad - 0.5) and (radius ** 2 < (crad - 0.5) ** 2 + 0.25):
            m1 = np.sqrt(radius ** 2 - (crad - 0.5) ** 2, dtype=np.complex)
            m1n = m1 / radius
            sg0 = 2 * (radius ** 2 * (0.5 * np.arcsin(m1n) + 0.25 * np.sin(2 * np.arcsin(m1n))) - m1 * (crad - 0.5))
            sgrid[2 * crad + 1, crad + 1] = sg0
            sgrid[crad + 1, 2 * crad + 1] = sg0
            sgrid[crad + 1, 1] = sg0
            sgrid[1, crad + 1] = sg0
            sgrid[2 * crad, crad + 1] = sgrid[2 * crad, crad + 1] - sg0
            sgrid[crad + 1, 2 * crad] = sgrid[crad + 1, 2 * crad] - sg0
            sgrid[crad + 1, 2] = sgrid[crad + 1, 2] - sg0
            sgrid[2, crad + 1] = sgrid[2, crad + 1] - sg0

        sgrid[crad + 1, crad + 1] = min(sgrid[crad + 1, crad + 1], 1)
        h = sgrid / np.sum(sgrid[:])
        return h.real


if __name__ == "__main__":
    kernel = fspecial('disk', 5)
    print(kernel)
