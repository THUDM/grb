"""Torch module for MLP."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    r"""

    Description
    -----------
    Multi-Layer Perceptron (without considering the graph structure).

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
        Activation function. Default: ``torch.nn.functional.relu``.
    dropout : float, optional
        Dropout rate during training. Default: ``0.0``.

    """

    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 n_layers,
                 activation=F.relu,
                 feat_norm=None,
                 adj_norm_func=None,
                 batch_norm=False,
                 layer_norm=False,
                 dropout=0.0):
        super(MLP, self).__init__()
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
            self.layers.append(MLPLayer(in_features=n_features[i],
                                        out_features=n_features[i + 1],
                                        activation=activation if i != n_layers - 1 else None,
                                        batch_norm=batch_norm if i != n_layers - 1 else None,
                                        dropout=dropout if i != n_layers - 1 else 0.0))
        self.reset_parameters()

    @property
    def model_type(self):
        """Indicate type of implementation."""
        return "torch"

    @property
    def model_name(self):
        return "mlp"

    def reset_parameters(self):
        """Reset parameters."""
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj=None):
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


class MLPLayer(nn.Module):
    r"""

    Description
    -----------
    MLP layer.

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``None``.
    dropout : float, optional
        Rate of dropout. Default: ``0.0``.

    """

    def __init__(self, in_features, out_features, activation=None, batch_norm=False, dropout=0.0):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_features)
        else:
            self.batch_norm = None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        if self.activation == F.leaky_relu:
            gain = nn.init.calculate_gain('leaky_relu')
        else:
            gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

    def forward(self, x, adj=None):
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

        x = self.linear(x)
        if hasattr(self, 'batch_norm'):
            if self.batch_norm is not None:
                x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)

        return x
