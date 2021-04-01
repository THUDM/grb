import torch
import numpy as np


def eval_acc(pred, labels, mask=None):
    if mask is None:
        return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)
    else:
        return (torch.argmax(pred[mask], dim=1) == labels[mask]).float().sum() / np.sum(mask)
