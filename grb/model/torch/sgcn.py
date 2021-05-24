import torch
import torch.nn as nn
import torch.nn.functional as F


class SGConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(SGConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj, k=4):
        for i in range(k):
            x = torch.spmm(adj, x)
        x = self.linear(x)

        return x


class SGCN(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, activation=torch.tanh, layer_norm=False):
        super(SGCN, self).__init__()
        if type(hidden_features) is int:
            hidden_features = [hidden_features]
        self.batchnorm = nn.BatchNorm1d(in_features)
        self.in_conv = nn.Linear(in_features, hidden_features[0])
        self.out_conv = nn.Linear(hidden_features[-1], out_features)
        self.activation = activation
        self.layers = nn.ModuleList()

        for i in range(len(hidden_features) - 1):
            if layer_norm:
                self.layers.append(nn.LayerNorm(hidden_features[i]))
            self.layers.append(SGConv(hidden_features[i], hidden_features[i + 1]))

    @property
    def model_type(self):
        return "torch"

    def forward(self, x, adj, dropout=0):
        x = self.batchnorm(x)
        x = self.in_conv(x)
        x = self.activation(x)
        x = F.dropout(x, dropout)

        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = layer(x, adj)
                x = self.activation(x)
        x = F.dropout(x, dropout)
        x = self.out_conv(x)

        return x
