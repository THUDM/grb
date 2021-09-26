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
    n_layers : int
        Number of layers.
    n_mlp_layers : int
        Number of layers.
    layer_norm : bool, optional
        Whether to use layer normalization. Default: ``False``.
    batch_norm : bool, optional
        Whether to apply batch normalization. Default: ``True``.
    eps : float, optional
        Hyper-parameter, refer to original paper. Default: ``0.0``.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``torch.nn.functional.relu``.
    feat_norm : str, optional
        Type of features normalization, choose from ["arctan", "tanh", None]. Default: ``None``.
    adj_norm_func : func of utils.normalize, optional
        Function that normalizes adjacency matrix. Default: ``None``.
    dropout : float, optional
            Rate of dropout. Default: ``0.0``.

    """

    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 n_layers,
                 n_mlp_layers=2,
                 activation=F.relu,
                 layer_norm=False,
                 batch_norm=True,
                 eps=0.0,
                 feat_norm=None,
                 adj_norm_func=None,
                 dropout=0.0):
        super(GIN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        self.activation = activation
        if type(hidden_features) is int:
            hidden_features = [hidden_features] * (n_layers - 1)
        elif type(hidden_features) is list or type(hidden_features) is tuple:
            assert len(hidden_features) == (n_layers - 1), "Incompatible sizes between hidden_features and n_layers."
        n_features = [in_features] + hidden_features + [out_features]

        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            if layer_norm:
                self.layers.append(nn.LayerNorm(n_features[i]))
            self.layers.append(GINConv(in_features=n_features[i],
                                       out_features=n_features[i + 1],
                                       batch_norm=batch_norm,
                                       eps=eps,
                                       activation=activation,
                                       dropout=dropout))
        self.mlp_layers = nn.ModuleList()
        for i in range(n_mlp_layers):
            if i == n_mlp_layers - 1:
                self.mlp_layers.append(nn.Linear(hidden_features[-1], out_features))
            else:
                self.mlp_layers.append(nn.Linear(hidden_features[-1], hidden_features[-1]))
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.reset_parameters()

    @property
    def model_type(self):
        """Indicate type of implementation."""
        return "torch"

    @property
    def model_name(self):
        return "gin"

    def reset_parameters(self):
        """Reset parameters."""
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.mlp_layers:
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

        for i, layer in enumerate(self.mlp_layers):
            x = layer(x)
            if i != len(self.mlp_layers) - 1:
                x = self.activation(x)
                if self.dropout is not None:
                    x = self.dropout(x)

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
    batch_norm : bool, optional
        Whether to apply batch normalization. Default: ``True``.
    dropout : float, optional
            Rate of dropout. Default: ``0.0``.

    """

    def __init__(self,
                 in_features,
                 out_features,
                 activation=F.relu,
                 eps=0.0,
                 batch_norm=True,
                 dropout=0.0):
        super(GINConv, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.activation = activation
        self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        self.batch_norm = batch_norm
        if batch_norm:
            self.norm = nn.BatchNorm1d(out_features)
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        if self.activation == F.leaky_relu:
            gain = nn.init.calculate_gain('leaky_relu')
        else:
            gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear1.weight, gain=gain)
        nn.init.xavier_normal_(self.linear2.weight, gain=gain)

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

        y = torch.spmm(adj, x)
        x = y + (1 + self.eps) * x
        x = self.linear1(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.linear2(x)
        if self.batch_norm:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)

        return x
