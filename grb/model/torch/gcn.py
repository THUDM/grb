import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, activation=None, dropout=False):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        if self.activation == F.leaky_relu:
            gain = nn.init.calculate_gain('leaky_relu')
        else:
            gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

    def forward(self, x, adj, dropout=0):
        x = self.linear(x)
        x = torch.spmm(adj, x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout:
            x = F.dropout(x, dropout)

        return x


class GCN(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, activation=F.relu, layer_norm=False, dropout=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if type(hidden_features) is int:
            hidden_features = [hidden_features]

        self.layers = nn.ModuleList()
        if layer_norm:
            self.layers.append(nn.LayerNorm(in_features))
        self.layers.append(GCNConv(in_features, hidden_features[0], activation=activation, dropout=dropout))
        for i in range(len(hidden_features) - 1):
            if layer_norm:
                self.layers.append(nn.LayerNorm(hidden_features[i]))
            self.layers.append(
                GCNConv(hidden_features[i], hidden_features[i + 1], activation=activation, dropout=dropout))
        self.layers.append(GCNConv(hidden_features[-1], out_features))
        self.reset_parameters()

    @property
    def model_type(self):
        return "torch"

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj, dropout=0):
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = layer(x, adj, dropout=dropout)

        return x
