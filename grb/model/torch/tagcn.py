"""Torch module for TAGCN."""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    k : int
        Hyper-parameter, refer to original paper.
    layer_norm : bool, optional
        Whether to use layer normalization. Default: ``False``.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``torch.nn.functional.leaky_relu``.
    dropout : bool, optional
        Whether to dropout during training. Default: ``True``.

    """

    def __init__(self, in_features, out_features, hidden_features, k, activation=F.leaky_relu,
                 layer_norm=False, dropout=True):
        super(TAGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if type(hidden_features) is int:
            hidden_features = [hidden_features]

        self.layers = nn.ModuleList()
        if layer_norm:
            self.layers.append(nn.LayerNorm(in_features))

        self.layers.append(TAGConv(in_features, hidden_features[0], k, activation=activation, dropout=dropout))
        for i in range(len(hidden_features) - 1):
            if layer_norm:
                self.layers.append(nn.LayerNorm(hidden_features[i]))
            self.layers.append(
                TAGConv(hidden_features[i], hidden_features[i + 1], k, activation=activation, dropout=dropout))
        self.layers.append(TAGConv(hidden_features[-1], out_features, k))
        self.reset_parameters()

    @property
    def model_type(self):
        """Indicate type of implementation."""
        return "torch"

    def reset_parameters(self):
        """Reset paramters."""
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj, dropout=0.0):
        r"""

        Parameters
        ----------
        x : torch.Tensor
            Tensor of input features.
        adj : list of torch.SparseTensor
            List of sparse tensor of adjacency matrix.
        dropout : float, optional
            Rate of dropout. Default: ``0.0``.

        Returns
        -------
        x : torch.Tensor
            Output of model (logits without activation).

        """

        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = layer(x, adj, dropout=dropout)

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
    dropout : bool, optional
        Whether to dropout during training. Default: ``False``.
    batchnorm : bool, optional
        Whether to apply batch normalization. Default: ``False``.

    """

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
        """Reset parameters."""
        if self.activation == F.leaky_relu:
            gain = nn.init.calculate_gain('leaky_relu')
        else:
            gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

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
            Output of layer.

        """

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
