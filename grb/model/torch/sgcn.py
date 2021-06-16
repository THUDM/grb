"""Torch module for SGCN."""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    layer_norm : bool, optional
        Whether to use layer normalization. Default: ``False``.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``torch.tanh``.

    """

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

    Returns
    -------
    x : torch.Tensor
        Output of layer.

    """

    def __init__(self, in_features, out_features):
        super(SGConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj, k=4):
        r"""

        Parameters
        ----------
        x : torch.Tensor
            Tensor of input features.
        adj : torch.SparseTensor
            Sparse tensor of adjacency matrix.
        k : int, optional
            Hyper-parameter, refer to original paper. Default: ``4``.

        Returns
        -------
        x : torch.Tensor
            Output of layer.

        """

        for i in range(k):
            x = torch.spmm(adj, x)
        x = self.linear(x)

        return x
