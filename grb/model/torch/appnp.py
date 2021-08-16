"""Torch module for APPNP."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from grb.utils.normalize import GCNAdjNorm


class APPNP(nn.Module):
    r"""

    Description
    -----------
    Approximated Personalized Propagation of Neural Predictions (`APPNP <https://arxiv.org/abs/1810.05997>`__)

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    hidden_features : int or list of int
        Dimension of hidden features. List if multi-layer.
    layer_norm : bool, optional
        Whether to use layer normalization. Default: ``False``.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``torch.nn.functional.relu``.
    feat_norm : str, optional
        Type of features normalization, choose from ["arctan", "tanh", None]. Default: ``None``.
    adj_norm_func : func of utils.normalize, optional
        Function that normalizes adjacency matrix. Default: ``GCNAdjNorm``.
    edge_drop : float, optional
        Rate of edge drop.
    alpha : float, optional
        Hyper-parameter, refer to original paper. Default: ``0.01``.
    k : int, optional
        Hyper-parameter, refer to original paper. Default: ``10``.
    dropout : float, optional
        Dropout rate during training. Default: ``0.0``.

    """

    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 layer_norm=False,
                 activation=F.relu,
                 edge_drop=0.0,
                 alpha=0.01,
                 k=10,
                 feat_norm=None,
                 adj_norm_func=GCNAdjNorm,
                 dropout=0.0):
        super(APPNP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        if type(hidden_features) is int:
            hidden_features = [hidden_features]
        self.layers = nn.ModuleList()
        if layer_norm:
            self.layers.append(nn.LayerNorm(in_features))
        self.layers.append(nn.Linear(in_features, hidden_features[0]))
        for i in range(len(hidden_features) - 1):
            if layer_norm:
                self.layers.append(nn.LayerNorm(hidden_features[i]))
            self.layers.append(nn.Linear(hidden_features[i], hidden_features[i+1]))
        self.layers.append(nn.Linear(hidden_features[-1], out_features))
        self.alpha = alpha
        self.k = k
        self.activation = activation
        if edge_drop > 0.0:
            self.edge_dropout = SparseEdgeDrop(edge_drop)
        else:
            self.edge_dropout = None
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    @property
    def model_type(self):
        """Indicate type of implementation."""
        return "torch"

    def reset_parameters(self):
        """Reset parameters."""
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj):
        r"""

        Parameters
        ----------
        x : torch.Tensor
            Tensor of input features.
        adj : torch.SparseTensor
            Sparse tensor of adjacency matrix.

        Returns
        -------
        x : torch.Tensor
            Output of model (logits without activation).

        """

        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = layer(x)
                x = self.activation(x)
                if self.dropout is not None:
                    x = self.dropout(x)
        for i in range(self.k):
            if self.edge_dropout is not None and self.training:
                adj = self.edge_dropout(adj)
            x = (1 - self.alpha) * torch.spmm(adj, x) + self.alpha * x

        return x


class SparseEdgeDrop(nn.Module):
    r"""

    Description
    -----------
    Sparse implementation of edge drop.

    Parameters
    ----------
    edge_drop : float
        Rate of edge drop.

    """

    def __init__(self, edge_drop):
        super(SparseEdgeDrop, self).__init__()
        self.edge_drop = edge_drop

    def forward(self, adj):
        """Sparse edge drop"""
        mask = ((torch.rand(adj._values().size()) + self.edge_drop) > 1.0)
        rc = adj._indices()
        val = adj._values().clone()
        val[mask] = 0.0

        return torch.sparse.FloatTensor(rc, val)
