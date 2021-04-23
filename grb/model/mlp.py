import torch.nn as nn
import torch.nn.functional as F


class MLPLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=None, dropout=False):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.dropout = dropout

    def forward(self, x, dropout=0):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout:
            x = F.dropout(x, dropout)

        return x


class MLP(nn.Module):
    def __init__(self, num_layers, num_features):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i != num_layers - 1:
                self.layers.append(
                    MLPLayer(num_features[i], num_features[i + 1], activation=F.relu, dropout=True))
            else:
                self.layers.append(MLPLayer(num_features[i], num_features[i + 1]))

    def forward(self, x, adj, dropout=0):
        for layer in self.layers:
            x = layer(x, adj, dropout=dropout)
        x = F.softmax(x, dim=-1)

        return x
