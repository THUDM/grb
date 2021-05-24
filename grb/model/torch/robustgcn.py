import torch
import torch.nn as nn
import torch.nn.functional as F


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

    @property
    def model_type(self):
        return "torch"

    def forward(self, x, adj, dropout=0):
        adj0, adj1 = adj
        mean = x
        var = x
        for layer in self.layers:
            mean, var = layer(mean, var=var, adj0=adj0, adj1=adj1, dropout=dropout)
        sample = torch.randn(var.shape).to(x.device)
        output = mean + sample * torch.pow(var, 0.5)

        return output
