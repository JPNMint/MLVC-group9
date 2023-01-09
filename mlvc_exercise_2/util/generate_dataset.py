from random import randint

import cv2
import numpy as np


def make_square():
    img = np.zeros((128, 128), dtype=np.uint8)

    ulx = randint(1, 128 - 50)
    uly = randint(1, 128 - 50)
    lrx = randint(ulx + 50, 128)
    lry = randint(uly + 50, 128)

    img = cv2.rectangle(img, (ulx, uly), (lrx, lry), (255), -1)

    return np.array(img)


def make_circle():
    img = np.zeros((128, 128), dtype=np.uint8)

    ulx = randint(1, 128 - 50)
    uly = randint(1, 128 - 50)
    l = 128 - (ulx if ulx > uly else uly)
    l = randint(max(50, l), l)

    img = cv2.circle(img, (int(ulx + l / 2), int(uly + l / 2)), int(l / 2),
                     (255), -1)

    return np.array(img)


def make_dataset(size):
    dataset = []
    labels = []
    for _ in range(size):
        dataset.append(make_square().reshape(128 * 128, ))
        labels.append(1)
        dataset.append(make_circle().reshape(128 * 128, ))
        labels.append(-1)

    return np.stack(dataset, axis=0), np.stack(labels)
