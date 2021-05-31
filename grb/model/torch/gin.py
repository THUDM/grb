import torch
import torch.nn as nn
import torch.nn.functional as F


class GINConv(nn.Module):
    def __init__(self, in_features, out_features, activation=F.relu, eps=0, batchnorm=True, dropout=False):
        super(GINConv, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.activation = activation
        self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        self.batchnorm = batchnorm
        if batchnorm:
            self.norm = nn.BatchNorm1d(out_features)
        self.dropout = dropout

    def reset_parameters(self):
        if self.activation == F.leaky_relu:
            gain = nn.init.calculate_gain('leaky_relu')
        else:
            gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weights, gain=gain)

    def forward(self, x, adj, dropout=0):
        y = torch.spmm(adj, x)
        x = y + (1 + self.eps) * x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        if self.batchnorm:
            x = self.norm(x)
        x = self.activation(x)
        if self.dropout:
            x = F.dropout(x, dropout)

        return x


class GIN(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, activation=F.relu, layer_norm=False, dropout=True):
        super(GIN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if type(hidden_features) is int:
            hidden_features = [hidden_features]
        self.layers = nn.ModuleList()

        if layer_norm:
            self.layers.append(nn.LayerNorm(in_features))

        self.layers.append(GINConv(in_features, hidden_features[0], activation=activation, dropout=dropout))
        for i in range(len(hidden_features) - 1):
            if layer_norm:
                self.layers.append(nn.LayerNorm(hidden_features[i]))
            self.layers.append(
                GINConv(hidden_features[i], hidden_features[i + 1], activation=activation))
        self.linear1 = nn.Linear(hidden_features[-2], hidden_features[-1])
        self.linear2 = nn.Linear(hidden_features[-1], out_features)

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

        x = F.relu(self.linear1(x))
        x = F.dropout(x, dropout)
        x = self.linear2(x)

        return x
