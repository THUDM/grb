import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseEdgeDrop(nn.Module):
    def __init__(self, edge_drop):
        super(SparseEdgeDrop, self).__init__()
        self.edge_drop = edge_drop

    def forward(self, x):
        mask = ((torch.rand(x._values().size()) + (1 - self.edge_drop)).floor()).type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / (1.0 - self.edge_drop + 1e-5))

        return torch.sparse.FloatTensor(rc, val)


class APPNP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, activation=F.relu, edge_drop=0, alpha=0.01, k=10):
        super(APPNP, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.alpha = alpha
        self.k = k
        self.edge_drop = edge_drop
        self.dropout = SparseEdgeDrop(edge_drop)
        self.activation = activation

    def forward(self, x, adj, dropout=0):
        x = self.linear1(x)
        x = self.activation(x)
        x = F.dropout(x, dropout)

        x = self.linear2(x)
        x = F.dropout(x, dropout)
        for i in range(self.k):
            if self.edge_drop != 0:
                adj = self.dropout(adj, dropout_prob=self.edge_drop)
            x = (1 - self.alpha) * torch.spmm(adj, x) + self.alpha * x

        return x
