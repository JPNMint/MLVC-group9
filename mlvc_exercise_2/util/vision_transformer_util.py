import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms


class CircleSquareDataset(Dataset):

    def __init__(self, data_X, data_y):
        X_tensor, y_tensor = data_X.astype(float) / 255, data_y.astype(float)
        X_tensor = np.reshape(X_tensor, (X_tensor.shape[0], 128, 128))

        tensors = (X_tensor, y_tensor)
        self.tensors = tensors
        self.transform = train_transforms = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        x = Image.fromarray(np.uint8((x) * 255))

        x = self.transform(x)

        x = x.repeat(3, 1, 1)

        # # hot-one encoding
        # if y == 1:
        #   label = np.array([0, 1])
        # else:
        #   label = np.array([1, 0])

        # Convert to classes
        if y == -1:
            label = 0
        if y == 1:
            label = 1

        return x, torch.from_numpy(np.array(label, dtype=np.float32)).float()

    def __len__(self):
        return self.tensors[0].shape[0]


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
