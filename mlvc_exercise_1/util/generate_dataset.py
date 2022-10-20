from random import randint

import cv2
import numpy as np


def make_square():
    img = np.zeros((128,128), dtype=np.uint8)

    ulx = randint(0, 118)
    uly = randint(0, 118)
    lrx = randint(ulx+10,128)
    lry = randint(uly+10,128)

    img = cv2.rectangle(img,(ulx,uly),(lrx,lry),(255),-1)

    return np.array(img)

def make_circle():
    img = np.zeros((128,128), dtype=np.uint8)

    ulx = randint(10, 118)
    uly = randint(10, 118)
    l = randint(10, 128)

    img = cv2.circle(img, (int(ulx),int(uly)), int(l/2), (255), -1)

    return np.array(img)

def make_dataset(num_samples, split):
    dataset = []
    labels = []
    for _ in range(int(num_samples/2)):
        dataset.append(make_square().reshape(128*128,))
        labels.append(1)
        dataset.append(make_circle().reshape(128*128,))
        labels.append(-1)
    
    dataset_np = np.stack(dataset, axis=0)
    labels_np =np.stack(labels)
    
    dataset_train = dataset_np[:int(num_samples*split)]
    labels_train = labels_np[:int(num_samples*split)]

    dataset_val = dataset_np[int(num_samples*split):]
    labels_val = labels_np[int(num_samples*split):]

    return (dataset_train, labels_train), (dataset_val, labels_val)
