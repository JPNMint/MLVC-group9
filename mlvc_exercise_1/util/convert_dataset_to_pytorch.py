import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset
from torchinfo import summary
from tqdm import tqdm, trange


class CircleSquareDataset(Dataset):
  def __init__(self, data_X, data_y):
    X_tensor, y_tensor = torch.tensor(data_X).float()/255, torch.tensor(data_y).float()
    tensors = (X_tensor, y_tensor)
    self.tensors = tensors

  def __getitem__(self, index):
    x = self.tensors[0][index]
    y = self.tensors[1][index]
    # Convert from -1 and 1 to 0 and 1
    y = torch.maximum(y, torch.zeros_like(y))

    return x, y

  def __len__(self):
    return self.tensors[0].size(0)

def convert_dataset_to_pytorch(dataset_train, labels_train, dataset_val, label_val, batch_size=128):
    dataset_pytorch_train = CircleSquareDataset(dataset_train, labels_train)
    dataset_pytorch_val = CircleSquareDataset(dataset_train, labels_train)

    train_dataloader = data.DataLoader(dataset_pytorch_train,
                                    shuffle=True,
                                    batch_size=batch_size)

    val_dataloader = data.DataLoader(dataset_pytorch_val,
                                    shuffle=True,
                                    batch_size=batch_size)

    return train_dataloader, val_dataloader
