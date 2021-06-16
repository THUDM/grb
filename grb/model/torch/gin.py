"""Torch module for GIN."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GIN(nn.Module):
    r"""

    Description
    -----------
    Graph Isomorphism Network (`GIN <https://arxiv.org/abs/1810.00826>`__)

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
        super(GIN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if type(hidden_features) is int:
            hidden_features = [hidden_features]
        self.layers = nn.ModuleList()

        if layer_norm:
            self.layers.append(nn.LayerNorm(in_features))

        self.layers.append(GINConv(in_features, hidden_features[0], activation=activation, dropout=dropout))
        for i in range(len(hidden_features) - 1):
            if layer_norm:
                self.layers.append(nn.LayerNorm(hidden_features[i]))
            self.layers.append(
                GINConv(hidden_features[i], hidden_features[i + 1], activation=activation))
        self.linear1 = nn.Linear(hidden_features[-2], hidden_features[-1])
        self.linear2 = nn.Linear(hidden_features[-1], out_features)

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

        x = F.relu(self.linear1(x))
        x = F.dropout(x, dropout)
        x = self.linear2(x)

        return x


class GINConv(nn.Module):
    r"""

    Description
    -----------
    GIN convolutional layer.

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``None``.
    eps : float, optional
        Hyper-parameter, refer to original paper. Default: ``0.0``.
    batchnorm : bool, optional
        Whether to apply batch normalization. Default: ``True``.
    dropout : bool, optional
        Whether to dropout during training. Default: ``False``.

    """

    def __init__(self, in_features, out_features, activation=F.relu, eps=0.0, batchnorm=True, dropout=False):
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
        """Reset parameters."""
        if self.activation == F.leaky_relu:
            gain = nn.init.calculate_gain('leaky_relu')
        else:
            gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weights, gain=gain)

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

        y = torch.spmm(adj, x)
        x = y + (1 + self.eps) * x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        if self.batchnorm:
            x = self.norm(x)
        x = self.activation(x)
        if self.dropout:
            x = F.dropout(x, dropout)

        return x
