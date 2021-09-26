import torch


def consistency_loss(logits, temp, lam):
    ps = [torch.exp(p) for p in logits]
    ps = torch.stack(ps, dim=2)

    avg_p = torch.mean(ps, dim=2)
    sharp_p = (torch.pow(avg_p, 1. / temp) / torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()

    sharp_p = sharp_p.unsqueeze(2)
    loss = torch.mean(torch.sum(torch.pow(ps - sharp_p, 2), dim=1, keepdim=True))

    loss = lam * loss

    return loss
