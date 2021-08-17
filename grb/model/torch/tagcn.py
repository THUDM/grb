"""Torch module for TAGCN."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from grb.utils.normalize import GCNAdjNorm


class TAGCN(nn.Module):
    r"""

    Description
    -----------
    Topological Adaptive Graph Convolutional Networks (`TAGCN <https://arxiv.org/abs/1710.10370>`__)

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
    k : int
        Hyper-parameter, k-hop adjacency matrix, refer to original paper.
    layer_norm : bool, optional
        Whether to use layer normalization. Default: ``False``.
    batch_norm : bool, optional
        Whether to apply batch normalization. Default: ``False``.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``torch.nn.functional.leaky_relu``.
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
                 k,
                 activation=F.leaky_relu,
                 feat_norm=None,
                 adj_norm_func=GCNAdjNorm,
                 layer_norm=False,
                 batch_norm=False,
                 dropout=0.0):
        super(TAGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        if type(hidden_features) is int:
            hidden_features = [hidden_features] * (n_layers - 1)
        elif type(hidden_features) is list or type(hidden_features) is tuple:
            assert len(hidden_features) == (n_layers - 1), "Incompatible sizes between hidden_features and n_layers."
        n_features = [in_features] + hidden_features + [out_features]

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if layer_norm:
                self.layers.append(nn.LayerNorm(n_features[i]))
            self.layers.append(TAGConv(in_features=n_features[i],
                                       out_features=n_features[i + 1],
                                       k=k,
                                       batch_norm=batch_norm if i != n_layers - 1 else False,
                                       activation=activation if i != n_layers - 1 else None,
                                       dropout=dropout if i != n_layers - 1 else 0.0))
        self.reset_parameters()

    @property
    def model_type(self):
        """Indicate type of implementation."""
        return "torch"

    def reset_parameters(self):
        """Reset paramters."""
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
                x = layer(x, adj)

        return x


class TAGConv(nn.Module):
    r"""

    Description
    -----------
    TAGCN convolutional layer.

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    k : int, optional
        Hyper-parameter, refer to original paper. Default: ``2``.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``None``.

    batch_norm : bool, optional
        Whether to apply batch normalization. Default: ``False``.
    dropout : float, optional
        Rate of dropout. Default: ``0.0``.

    """

    def __init__(self,
                 in_features,
                 out_features,
                 k=2,
                 activation=None,
                 batch_norm=False,
                 dropout=0.0):
        super(TAGConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features * (k + 1), out_features)
        self.batch_norm = batch_norm
        if batch_norm:
            self.norm_func = nn.BatchNorm1d(out_features, affine=False)
        self.activation = activation
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.k = k
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        if self.activation == F.leaky_relu:
            gain = nn.init.calculate_gain('leaky_relu')
        else:
            gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

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

        fstack = [x]
        for i in range(self.k):
            y = torch.spmm(adj, fstack[-1])
            fstack.append(y)
        x = torch.cat(fstack, dim=-1)
        x = self.linear(x)
        if self.batch_norm:
            x = self.norm_func(x)
        if not (self.activation is None):
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)

        return x
