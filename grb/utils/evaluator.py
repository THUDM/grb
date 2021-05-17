import torch
import numpy as np


def eval_acc(pred, labels, mask=None):
    if mask is None:
        return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)
    else:
        if torch.sum(mask) != 0:
            return (torch.argmax(pred[mask], dim=1) == labels[mask]).float().sum() / int(torch.sum(mask))
        else:
            return 0.0


def get_weights_arithmetic(n, w_1):
    weights = []
    epsilon = 2 / (n - 1) * (1 / n - w_1)
    for i in range(1, n + 1):
        weights.append(w_1 + (n - 1) * epsilon)
    weights = np.sort(weights)[::-1]

    return weights


def get_weights_sqrt(n, w_1):
    weights = []
    epsilon = 2 / (n - 1) * (1 / n - w_1)
    for i in range(1, n + 1):
        weights.append(w_1 + (n - 1) * epsilon)
    weights = np.sort(weights)[::-1]

    return weights
