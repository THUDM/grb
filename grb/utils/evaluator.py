import torch


def eval_acc(pred, labels, mask=None):
    if mask is None:
        return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)
    else:
        if torch.sum(mask) != 0:
            return (torch.argmax(pred[mask], dim=1) == labels[mask]).float().sum() / int(torch.sum(mask))
        else:
            return 0.0
