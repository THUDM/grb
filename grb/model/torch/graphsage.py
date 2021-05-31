import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGEConv(nn.Module):
    def __init__(self, in_features, pool_features, out_features, activation=None, dropout=False):
        super(SAGEConv, self).__init__()
        self.pool_fc = nn.Linear(in_features, pool_features)
        self.fc1 = nn.Linear(pool_features, out_features)
        self.fc2 = nn.Linear(pool_features, out_features)
        self.activation = activation
        self.dropout = dropout

    def forward(self, x, adj, dropout=0, mu=2.0):

        x = F.relu(self.pool_fc(x))
        x3 = x ** mu
        x2 = torch.spmm(adj, x3) ** (1 / mu)

        # In original model this is actually max-pool, but **10/**0.1 result in graident explosion.
        # However we can still achieve similar performance using 2-norm.
        x4 = self.fc1(x)
        x2 = self.fc2(x2)
        x4 = x4 + x2
        if self.activation is not None:
            x4 = self.activation(x4)
        if self.dropout:
            x4 = F.dropout(x4, dropout)

        return x4


class GraphSAGE(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, activation=F.relu, layer_norm=False, dropout=True):
        super(GraphSAGE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        if type(hidden_features) is int:
            hidden_features = [hidden_features]

        self.layers = nn.ModuleList()
        if layer_norm:
            self.layers.append(nn.LayerNorm(in_features))
        self.layers.append(SAGEConv(in_features, in_features, hidden_features[0],
                                    activation=activation, dropout=dropout))
        for i in range(len(hidden_features) - 1):
            if layer_norm:
                self.layers.append(nn.LayerNorm(hidden_features[i]))
            self.layers.append(SAGEConv(hidden_features[i], hidden_features[i],
                                        hidden_features[i + 1], activation=activation, dropout=dropout))
        self.layers.append(SAGEConv(hidden_features[-1], hidden_features[-1], out_features))

    @property
    def model_type(self):
        return "torch"

    def forward(self, x, adj, dropout=0):
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = F.normalize(x, dim=1)
                x = layer(x, adj, dropout=dropout)

        return x
