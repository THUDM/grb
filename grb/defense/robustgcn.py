"""Torch module for RobustGCN."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from grb.utils.normalize import RobustGCNAdjNorm

class RobustGCN(nn.Module):
    r"""

    Description
    -----------
    Robust Graph Convolutional Networks (`RobustGCN <http://pengcui.thumedialab.com/papers/RGCN.pdf>`__)

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    hidden_features : int or list of int
        Dimension of hidden features. List if multi-layer.
    feat_norm : str, optional
        Type of features normalization, choose from ["arctan", "tanh", None]. Default: ``None``.
    adj_norm_func : func of utils.normalize, optional
        Function that normalizes adjacency matrix. Default: ``RobustAdjNorm``.
    dropout : float, optional
            Rate of dropout. Default: ``0.0``.

    """

    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 n_layers,
                 feat_norm=None,
                 adj_norm_func=RobustGCNAdjNorm,
                 dropout=0.0):
        super(RobustGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        if type(hidden_features) is int:
            hidden_features = [hidden_features] * (n_layers - 1)
        elif type(hidden_features) is list or type(hidden_features) is tuple:
            assert len(hidden_features) == (n_layers - 1), "Incompatible sizes between hidden_features and n_layers."
        n_features = [in_features] + hidden_features + [out_features]

        self.act0 = F.elu
        self.act1 = F.relu

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(RobustGCNConv(n_features[i], n_features[i + 1], act0=self.act0, act1=self.act1,
                                             initial=True if i == 0 else False,
                                             dropout=dropout if i != n_layers - 1 else 0.0))

    @property
    def model_type(self):
        """Indicate type of implementation."""
        return "torch"

    def forward(self, x, adj):
        r"""

        Parameters
        ----------
        x : torch.Tensor
            Tensor of input features.
        adj : list of torch.SparseTensor
            List of sparse tensor of adjacency matrix.

        Returns
        -------
        x : torch.Tensor
            Output of model (logits without activation).

        """

        adj0, adj1 = adj
        mean = x
        var = x
        for layer in self.layers:
            mean, var = layer(mean, var=var, adj0=adj0, adj1=adj1)
        sample = torch.randn(var.shape).to(x.device)
        output = mean + sample * torch.pow(var, 0.5)

        return output


class RobustGCNConv(nn.Module):
    r"""

    Description
    -----------
    RobustGCN convolutional layer.

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    act0 : func of torch.nn.functional, optional
        Activation function. Default: ``F.elu``.
    act1 : func of torch.nn.functional, optional
        Activation function. Default: ``F.relu``.
    initial : bool, optional
        Whether to initialize variance.
    dropout : float, optional
            Rate of dropout. Default: ``0.0``.

    """

    def __init__(self, in_features, out_features, act0=F.elu, act1=F.relu, initial=False, dropout=0.0):
        super(RobustGCNConv, self).__init__()
        self.mean_conv = nn.Linear(in_features, out_features)
        self.var_conv = nn.Linear(in_features, out_features)
        self.act0 = act0
        self.act1 = act1
        self.initial = initial
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, mean, var=None, adj0=None, adj1=None):
        r"""

        Parameters
        ----------
        mean : torch.Tensor
            Tensor of mean of input features.
        var : torch.Tensor, optional
            Tensor of variance of input features. Default: ``None``.
        adj0 : torch.SparseTensor, optional
            Sparse tensor of adjacency matrix 0. Default: ``None``.
        adj1 : torch.SparseTensor, optional
            Sparse tensor of adjacency matrix 1. Default: ``None``.

        Returns
        -------

        """
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
            if self.dropout is not None:
                mean = self.dropout(mean)
                var = self.dropout(var)

        return mean, var
