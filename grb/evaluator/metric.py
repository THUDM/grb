import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def eval_acc(pred, labels, mask=None):
    if mask is not None:
        pred, labels = pred[mask], labels[mask]
        if pred is None or labels is None:
            return 0.0

    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def eval_rocauc(pred, labels, mask=None):
    rocauc_list = []
    if mask is not None:
        pred, labels = pred[mask], labels[mask]
        if pred is None or labels is None:
            return 0.0
    pred = pred.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    for i in range(labels.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(labels[:, i] == 1) > 0 and np.sum(labels[:, i] == 0) > 0:
            rocauc_list.append(roc_auc_score(y_true=labels[:, i],
                                             y_score=pred[:, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


def eval_f1multilabel(pred, labels, mask=None):
    if mask is not None:
        pred, labels = pred[mask], labels[mask]
        if pred is None or labels is None:
            return 0.0
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    tp = (labels * pred).sum().float()
    fp = ((1 - labels) * pred).sum().float()
    fn = (labels * (1 - pred)).sum().float()

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = (2 * precision * recall) / (precision + recall + epsilon)

    return f1.item()


def get_weights_arithmetic(n, w_1, order='a'):
    """

    :param n:
    :param w_1:
    :param order: 'a' ascending or 'd' descending.
    :return:
    """
    weights = []
    epsilon = 2 / (n - 1) * (1 / n - w_1)
    for i in range(1, n + 1):
        weights.append(w_1 + (i - 1) * epsilon)

    if 'd' in order:
        weights.reverse()

    return weights


def get_weights_polynomial(n, p=2, order='a'):
    """

    :param n:
    :param p:
    :param order: 'a' ascending or 'd' descending.
    :return:
    """
    weights = []
    for i in range(1, n + 1):
        weights.append(1 / i ** p)
    weights_norm = [weights[i] / sum(weights) for i in range(n)]
    if 'a' in order:
        weights_norm = weights_norm[::-1]

    return weights_norm
