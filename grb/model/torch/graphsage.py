"""Torch module for GraphSAGE."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGE(nn.Module):
    r"""

    Description
    -----------
    Inductive Representation Learning on Large Graphs (`GraphSAGE <https://arxiv.org/abs/1706.02216>`__)

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
        super(GraphSAGE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        if type(hidden_features) is int:
            hidden_features = [hidden_features]

        self.layers = nn.ModuleList()
        if layer_norm:
            self.layers.append(nn.LayerNorm(in_features))
        self.layers.append(SAGEConv(in_features, in_features, hidden_features[0],
                                    activation=activation, dropout=dropout))
        for i in range(len(hidden_features) - 1):
            if layer_norm:
                self.layers.append(nn.LayerNorm(hidden_features[i]))
            self.layers.append(SAGEConv(hidden_features[i], hidden_features[i],
                                        hidden_features[i + 1], activation=activation, dropout=dropout))
        self.layers.append(SAGEConv(hidden_features[-1], hidden_features[-1], out_features))

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

        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = F.normalize(x, dim=1)
                x = layer(x, adj, dropout=dropout)

        return x


class SAGEConv(nn.Module):
    r"""

    Description
    -----------
    SAGE convolutional layer.

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    pool_features : int
        Dimension of pooling features.
    out_features : int
        Dimension of output features.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``None``.
    dropout : bool, optional
        Whether to dropout during training. Default: ``False``.

    """

    def __init__(self, in_features, pool_features, out_features, activation=None, dropout=False):
        super(SAGEConv, self).__init__()
        self.pool_fc = nn.Linear(in_features, pool_features)
        self.fc1 = nn.Linear(pool_features, out_features)
        self.fc2 = nn.Linear(pool_features, out_features)
        self.activation = activation
        self.dropout = dropout

    def forward(self, x, adj, dropout=0.0, mu=2.0):
        r"""

        Parameters
        ----------
        x : torch.Tensor
            Tensor of input features.
        adj : torch.SparseTensor
            Sparse tensor of adjacency matrix.
        dropout : float, optional
            Rate of dropout. Default: ``0.0``.
        mu : float, optional
            Hyper-parameter, refer to original paper. Default: ``2.0``.

        Returns
        -------
        x : torch.Tensor
            Output of layer.

        """

        x = F.relu(self.pool_fc(x))
        x3 = x ** mu
        x2 = torch.spmm(adj, x3) ** (1 / mu)

        # In original model this is actually max-pool, but **10/**0.1 result in graident explosion.
        # However we can still achieve similar performance using 2-norm.
        x4 = self.fc1(x)
        x2 = self.fc2(x2)
        x4 = x4 + x2
        if self.activation is not None:
            x4 = self.activation(x4)
        if self.dropout:
            x4 = F.dropout(x4, dropout)

        return x4
