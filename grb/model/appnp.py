import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseDropout(torch.nn.Module):
    def __init__(self):
        super(SparseDropout, self).__init__()

    def forward(self, x, dropout_prob=0.5):
        mask = ((torch.rand(x._values().size()) + (1 - dropout_prob)).floor()).type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / (1.0 - dropout_prob + 1e-5))
        return torch.sparse.FloatTensor(rc, val)


class APPNP(nn.Module):
    def __init__(self, in_features, out_features, num_hidden, alpha=0.01, k=10):
        super(APPNP, self).__init__()
        self.linear1 = nn.Linear(in_features, num_hidden)
        self.linear2 = nn.Linear(num_hidden, out_features)
        self.alpha = alpha
        self.dropout = SparseDropout()
        self.activation = F.relu
        self.k = k

    def forward(self, x, adj, dropout=0):
        x = self.linear1(x)
        x = self.activation(x)
        x = F.dropout(x, dropout)

        x = self.linear2(x)
        x = F.dropout(x, dropout)
        for i in range(self.K):
            adj_drop = self.drop(adj, dprob=dropout)
            x = (1 - self.alpha) * torch.spmm(adj_drop, x) + self.alpha * x

        return x
