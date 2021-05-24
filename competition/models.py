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
    def __init__(self, in_features, out_features, hidden_features, activation=F.relu, dropout=True):
        super(GraphSAGE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if type(hidden_features) is int:
            hidden_features = [hidden_features]
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_features, in_features, hidden_features[0],
                                    activation=activation, dropout=dropout))
        for i in range(len(hidden_features) - 1):
            self.layers.append(SAGEConv(hidden_features[i], hidden_features[i],
                                        hidden_features[i + 1], activation=F.relu, dropout=dropout))
        self.layers.append(SAGEConv(hidden_features[-1], hidden_features[-1], out_features))

    def forward(self, x, adj, dropout=0):
        for layer in self.layers:
            x = F.normalize(x, dim=1)
            x = layer(x, adj, dropout=dropout)

        return x


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


class RobustGCNConv(nn.Module):
    def __init__(self, in_feats, out_feats, act0=F.elu, act1=F.relu, initial=False, dropout=False):
        super(RobustGCNConv, self).__init__()
        self.mean_conv = nn.Linear(in_feats, out_feats)
        self.var_conv = nn.Linear(in_feats, out_feats)
        self.act0 = act0
        self.act1 = act1
        self.initial = initial
        self.dropout = dropout

    def forward(self, mean, var=None, adj0=None, adj1=None, dropout=0):
        mean = self.mean_conv(mean)
        if self.initial:
            var = mean * 1
        else:
            var = self.var_conv(var)
        mean = self.act0(mean)
        var = self.act1(var)
        attention = torch.exp(-var)

        mean = mean * attention
        var = var * attention * attention
        mean = torch.spmm(adj0, mean)
        var = torch.spmm(adj1, var)
        if self.dropout:
            mean = self.act0(mean)
            var = self.act0(var)
            mean = F.dropout(mean, dropout)
            var = F.dropout(var, dropout)

        return mean, var


class RobustGCN(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, dropout=True):
        super(RobustGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if type(hidden_features) is int:
            hidden_features = [hidden_features]
        self.act0 = F.elu
        self.act1 = F.relu

        self.layers = nn.ModuleList()
        self.layers.append(RobustGCNConv(in_features, hidden_features[0], act0=self.act0, act1=self.act1,
                                         initial=True, dropout=dropout))
        for i in range(len(hidden_features) - 1):
            self.layers.append(RobustGCNConv(hidden_features[i], hidden_features[i + 1],
                                             act0=self.act0, act1=self.act1, dropout=True))
        self.layers.append(RobustGCNConv(hidden_features[-1], out_features, act0=self.act0, act1=self.act1))

    def forward(self, x, adj, dropout=0):
        adj0, adj1 = adj
        mean = x
        var = x
        for layer in self.layers:
            mean, var = layer(mean, var=var, adj0=adj0, adj1=adj1, dropout=dropout)
        sample = torch.randn(var.shape).to(x.device)
        output = mean + sample * torch.pow(var, 0.5)

        return output


class TAGConv(nn.Module):
    def __init__(self, in_features, out_features, k=2, activation=None, dropout=False, batchnorm=False):
        super(TAGConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features * (k + 1), out_features)
        self.batchnorm = batchnorm
        if batchnorm:
            self.norm_func = nn.BatchNorm1d(out_features, affine=False)
        self.activation = activation
        self.dropout = dropout
        self.k = k
        self.reset_parameters()

    def reset_parameters(self):
        if self.activation == F.leaky_relu:
            gain = nn.init.calculate_gain('leaky_relu')
        else:
            gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

    def forward(self, x, adj, dropout=0):

        fstack = [x]
        for i in range(self.k):
            y = torch.spmm(adj, fstack[-1])
            fstack.append(y)
        x = torch.cat(fstack, dim=-1)
        x = self.linear(x)
        if self.batchnorm:
            x = self.norm_func(x)
        if not (self.activation is None):
            x = self.activation(x)
        if self.dropout:
            x = F.dropout(x, dropout)

        return x


class TAGCN(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, k, activation=F.leaky_relu, dropout=True):
        super(TAGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if type(hidden_features) is int:
            hidden_features = [hidden_features]

        self.layers = nn.ModuleList()
        self.layers.append(TAGConv(in_features, hidden_features[0], k, activation=activation, dropout=dropout))
        for i in range(len(hidden_features) - 1):
            self.layers.append(
                TAGConv(hidden_features[i], hidden_features[i + 1], k, activation=activation, dropout=dropout))
        self.layers.append(TAGConv(hidden_features[-1], out_features, k))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj, dropout=0):
        for i in range(len(self.layers)):
            x = self.layers[i](x, adj, dropout=dropout)

        return x


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


class GINConv(nn.Module):
    def __init__(self, in_features, out_features, activation=F.relu, eps=0, batchnorm=False, dropout=False):
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
        if self.dropout:
            x = F.dropout(x, dropout)

        return x


class GIN(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, activation=F.relu, dropout=True):
        super(GIN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if type(hidden_features) is int:
            hidden_features = [hidden_features]
        self.layers = nn.ModuleList()

        self.layers.append(GINConv(in_features, hidden_features[0], activation=activation, dropout=dropout))
        for i in range(len(hidden_features) - 1):
            self.layers.append(
                GINConv(hidden_features[i], hidden_features[i + 1], activation=activation))
        self.linear1 = nn.Linear(hidden_features[-2], hidden_features[-1])
        self.linear2 = nn.Linear(hidden_features[-1], out_features)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj, dropout=0):
        for layer in self.layers:
            x = layer(x, adj, dropout=dropout)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, dropout)
        x = self.linear2(x)

        return x
