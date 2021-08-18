"""Torch module for SGCN."""
import torch
import torch.nn as nn

from grb.utils.normalize import GCNAdjNorm


class SGCN(nn.Module):
    r"""

    Description
    -----------
    Simplifying Graph Convolutional Networks (`SGCN <https://arxiv.org/abs/1902.07153>`__)

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    hidden_features : int or list of int
        Dimension of hidden features. List if multi-layer.
    n_layers : int
        Number of layers.
    layer_norm : bool, optional
        Whether to use layer normalization. Default: ``False``.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``torch.tanh``.
    k : int, optional
        Hyper-parameter, refer to original paper. Default: ``4``.
    feat_norm : str, optional
        Type of features normalization, choose from ["arctan", "tanh", None]. Default: ``None``.
    adj_norm_func : func of utils.normalize, optional
        Function that normalizes adjacency matrix. Default: ``GCNAdjNorm``.
    dropout : float, optional
            Rate of dropout. Default: ``0.0``.

    """

    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 n_layers,
                 activation=torch.tanh,
                 feat_norm=None,
                 adj_norm_func=GCNAdjNorm,
                 layer_norm=False,
                 k=4,
                 dropout=0.0):
        super(SGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        if type(hidden_features) is int:
            hidden_features = [hidden_features] * (n_layers - 1)
        elif type(hidden_features) is list or type(hidden_features) is tuple:
            assert len(hidden_features) == (n_layers - 1), "Incompatible sizes between hidden_features and n_layers."

        self.batch_norm = nn.BatchNorm1d(in_features)
        self.in_conv = nn.Linear(in_features, hidden_features[0])
        self.out_conv = nn.Linear(hidden_features[-1], out_features)
        self.activation = activation
        self.layers = nn.ModuleList()

        for i in range(n_layers - 2):
            if layer_norm:
                self.layers.append(nn.LayerNorm(hidden_features[i]))
            self.layers.append(SGConv(in_features=hidden_features[i],
                                      out_features=hidden_features[i + 1],
                                      k=k))

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

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
        adj : torch.SparseTensor
            Sparse tensor of adjacency matrix.

        Returns
        -------
        x : torch.Tensor
            Output of model (logits without activation).

        """

        x = self.batch_norm(x)
        x = self.in_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)

        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = layer(x, adj)
                if self.activation is not None:
                    x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.out_conv(x)

        return x


class SGConv(nn.Module):
    r"""

    Description
    -----------
    SGCN convolutional layer.

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    k : int, optional
        Hyper-parameter, refer to original paper. Default: ``4``.

    Returns
    -------
    x : torch.Tensor
        Output of layer.

    """

    def __init__(self, in_features, out_features, k):
        super(SGConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.k = k

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
            Output of layer.

        """

        for i in range(self.k):
            x = torch.spmm(adj, x)
        x = self.linear(x)

        return x
