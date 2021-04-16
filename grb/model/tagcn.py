import torch
import torch.nn as nn
import torch.nn.functional as F


class TAGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, k=2, activation=None, dropout=False, norm=False):
        super(TAGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features * (k + 1), out_features)
        self.norm = norm
        self.norm_func = nn.BatchNorm1d(out_features, affine=False)
        self.activation = activation
        self.dropout = dropout
        self.k = k
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

    def forward(self, x, adj, dropout=0):

        fstack = [x]

        for i in range(self.k):
            y = torch.spmm(adj, fstack[-1])
            fstack.append(y)
        x = torch.cat(fstack, dim=-1)
        x = self.linear(x)
        if self.norm:
            x = self.norm_func(x)
        if not (self.activation is None):
            x = self.activation(x)
        if self.dropout:
            x = F.dropout(x, dropout)

        return x


class TAGCN(nn.Module):
    def __init__(self, num_layers, num_features, k):
        super(TAGCN, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i != num_layers - 1:
                self.layers.append(
                    TAGraphConvolution(num_features[i], num_features[i + 1], k, activation=F.leaky_relu, dropout=True))
            else:
                self.layers.append(TAGraphConvolution(num_features[i], num_features[i + 1], k))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj, dropout=0):
        for i in range(len(self.layers)):
            x = self.layers[i](x, adj, dropout=dropout)

        return x
