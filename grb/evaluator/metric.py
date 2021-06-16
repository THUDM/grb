"""Evaluation metrics"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def eval_acc(pred, labels, mask=None):
    r"""

    Description
    -----------
    Accuracy metric for node classification.

    Parameters
    ----------
    pred : torch.Tensor
        Output logits of model in form of ``N * 1``.
    labels : torch.LongTensor
        Labels in form of ``N * 1``.
    mask : torch.Tensor, optional
        Mask of nodes to evaluate in form of ``N * 1`` torch bool tensor. Default: ``None``.

    Returns
    -------
    acc : float
        Node classification accuracy.

    """

    if mask is not None:
        pred, labels = pred[mask], labels[mask]
        if pred is None or labels is None:
            return 0.0

    acc = (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

    return acc


def eval_rocauc(pred, labels, mask=None):
    r"""

    Description
    -----------
    ROC-AUC score for multi-label node classification.

    Parameters
    ----------
    pred : torch.Tensor
        Output logits of model in form of ``N * 1``.
    labels : torch.LongTensor
        Labels in form of ``N * 1``.
    mask : torch.Tensor, optional
        Mask of nodes to evaluate in form of ``N * 1`` torch bool tensor. Default: ``None``.


    Returns
    -------
    rocauc : float
        Average ROC-AUC score across different labels.

    """

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

    rocauc = sum(rocauc_list) / len(rocauc_list)

    return rocauc


def eval_f1multilabel(pred, labels, mask=None):
    r"""

    Description
    -----------
    F1 score for multi-label node classification.

    Parameters
    ----------
    pred : torch.Tensor
        Output logits of model in form of ``N * 1``.
    labels : torch.LongTensor
        Labels in form of ``N * 1``.
    mask : torch.Tensor, optional
        Mask of nodes to evaluate in form of ``N * 1`` torch bool tensor. Default: ``None``.


    Returns
    -------
    f1 : float
        Average F1 score across different labels.

    """

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
    f1 = f1.item()

    return f1


def get_weights_arithmetic(n, w_1, order='a'):
    r"""

    Description
    -----------
    Arithmetic weights for calculating weighted robust score.

    Parameters
    ----------
    n : int
        Number of scores.
    w_1 : float
        Initial weight of the first term.
    order : str, optional
        ``a`` for ascending order, ``d`` for descending order. Default: ``a``.

    Returns
    -------
    weights : list
        List of weights.

    """

    weights = []
    epsilon = 2 / (n - 1) * (1 / n - w_1)
    for i in range(1, n + 1):
        weights.append(w_1 + (i - 1) * epsilon)

    if 'd' in order:
        weights.reverse()

    return weights


def get_weights_polynomial(n, p=2, order='a'):
    r"""

    Description
    -----------
    Arithmetic weights for calculating weighted robust score.

    Parameters
    ----------
    n : int
        Number of scores.
    p : float
        Power of denominator.
    order : str, optional
        ``a`` for ascending order, ``d`` for descending order. Default: ``a``.

    Returns
    -------
    weights_norms : list
        List of normalized polynomial weights.

    """

    weights = []
    for i in range(1, n + 1):
        weights.append(1 / i ** p)
    weights_norm = [weights[i] / sum(weights) for i in range(n)]
    if 'a' in order:
        weights_norm = weights_norm[::-1]

    return weights_norm
