import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset
from tqdm import tqdm, trange


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 10)
        self.hidden_fc = nn.Linear(10, 4)
        self.output_fc = nn.Linear(4, output_dim)

    def forward(self, x):
        
        # x = [batch size, height * width]

        h_1 = torch.sigmoid(self.input_fc(x))

        # h_1 = [batch size, 10]

        h_2 = torch.sigmoid(self.hidden_fc(h_1))

        # h_2 = [batch size, 4]

        y_pred = self.output_fc(h_2)

        # y_pred = [batch size, output dim]

        return torch.sigmoid(torch.squeeze(y_pred)), h_2

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_accuracy(y_pred, y):
    assert y.ndim == 1 and y.size() == y_pred.size()
    y_pred = y_pred > 0.5
    return (y == y_pred).sum().item() / y.size(0)

def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred, _ = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def vis_loss(loss_train_plot, loss_test_plot, acc_train_plot, acc_test_plot):
    epochs_nr = np.arange(0,len(loss_train_plot),1)+1
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=100)

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs_nr, loss_train_plot, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs_nr, loss_test_plot, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(which="both")
    ax1.set_ylim([0, 1.01])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs_nr, acc_train_plot, color=color, linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs_nr, acc_test_plot, color=color, linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 101])
    plt.legend(['Train', 'Test'], loc="center right")


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

def vis_weights(model):
    input_weight = model.input_fc.weight.cpu().detach().numpy()
    hidden_weight = model.hidden_fc.weight.cpu().detach().numpy()
    output_weight = model.output_fc.weight.cpu().detach().numpy()

    print(input_weight.shape, hidden_weight.shape, output_weight.shape)

    fig, axs = plt.subplots(2, 5, figsize=(5, 2), dpi=300)

    for i, ax in enumerate(axs.reshape(-1)):
        ax.imshow(input_weight[i,:].reshape((128, 128)))
        ax.axis('off')
    fig.tight_layout()
    plt.show()

    fig = plt.figure()
    plt.imshow(hidden_weight)
    plt.axis('off')
    plt.show()

    fig = plt.figure()
    plt.imshow(output_weight)
    plt.axis('off')
    plt.show()
