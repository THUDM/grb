import torch
import torch.nn as nn
import torch.nn.functional as F


class GINConv(nn.Module):
    def __init__(self, in_features, out_features, activation=F.relu, eps=0, batchnorm=True, dropout=True):
        super(GINConv, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.activation = activation
        self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        self.batchnorm = batchnorm
        if batchnorm:
            self.norm = nn.BatchNorm1d(out_features)
        self.dropout = dropout

    def forward(self, x, adj, dropout=0):
        y = torch.spmm(adj, x)
        x = y + (1 + self.eps) * x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        if self.batchnorm:
            x = self.norm(x)
        if self.dropout:
            x = F.dropout(x, dropout)

        return x


class GIN(nn.Module):
    def __init__(self, num_layers, num_features, activation=F.relu):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.layers = nn.ModuleList()

        for i in range(num_layers - 1):
            self.layers.append(
                GINConv(num_features[i], num_features[i + 1], activation=activation))
        self.linear1 = nn.Linear(num_features[-2], num_features[-2])
        self.linear2 = nn.Linear(num_features[-2], num_features[-1])

    def forward(self, x, adj, dropout=0):
        for layer in self.layers:
            x = layer(x, adj, dropout=dropout)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, dropout)
        x = self.linear2(x)

        return x
