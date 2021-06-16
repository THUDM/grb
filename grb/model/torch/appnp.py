"""Torch module for APPNP."""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    edge_drop : float, optional
        Rate of edge drop.
    alpha : float, optional
        Hyper-parameter, refer to original paper. Default: ``0.01``.
    k : int, optional
        Hyper-parameter, refer to original paper. Default: ``10``.

    """

    def __init__(self, in_features, out_features, hidden_features, layer_norm=False,
                 activation=F.relu, edge_drop=0.0, alpha=0.01, k=10):
        super(APPNP, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(in_features)
            self.layer_norm2 = nn.LayerNorm(hidden_features)
        self.alpha = alpha
        self.k = k
        self.edge_drop = edge_drop
        self.dropout = SparseEdgeDrop(edge_drop)
        self.activation = activation

    @property
    def model_type(self):
        """Indicate type of implementation."""
        return "torch"

    def forward(self, x, adj, dropout=0.0):
        r"""

        Parameters
        ----------
        x : torch.Tensor
            Tensor of input features.
        adj : torch.SparseTensor
            Sparse tensor of adjacency matrix.
        dropout : float, optional
            Rate of dropout. Default: ``0.0``.

        Returns
        -------
        x : torch.Tensor
            Output of model (logits without activation).

        """

        if self.layer_norm:
            x = self.layer_norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = F.dropout(x, dropout)
        if self.layer_norm:
            x = self.layer_norm2(x)
        x = self.linear2(x)
        x = F.dropout(x, dropout)
        for i in range(self.k):
            if self.edge_drop != 0:
                adj = self.dropout(adj, dropout_prob=self.edge_drop)
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

    def forward(self, x):
        """Sparse edge drop"""
        mask = ((torch.rand(x._values().size()) + (1 - self.edge_drop)).floor()).type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / (1.0 - self.edge_drop + 1e-5))

        return torch.sparse.FloatTensor(rc, val)
