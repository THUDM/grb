"""Torch module for GCN."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    r"""

    Description
    -----------
    Graph Convolutional Networks (`GCN <https://arxiv.org/abs/1609.02907>`__)

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
    dropout : bool, optional
        Whether to dropout during training. Default: ``True``.

    """

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

    @property
    def model_type(self):
        """Indicate type of implementation."""
        return "torch"

    def reset_parameters(self):
        """Reset parameters."""
        for layer in self.layers:
            layer.reset_parameters()

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

        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = layer(x, adj, dropout=dropout)

        return x


class GCNConv(nn.Module):
    r"""

    Description
    -----------
    GCN convolutional layer.

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``None``.
    dropout : bool, optional
        Whether to dropout during training. Default: ``False``.

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

        x = self.linear(x)
        x = torch.spmm(adj, x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout:
            x = F.dropout(x, dropout)

        return x
