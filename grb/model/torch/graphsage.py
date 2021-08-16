"""Torch module for GraphSAGE."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from grb.utils.normalize import SAGEAdjNorm


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
    feat_norm : str, optional
        Type of features normalization, choose from ["arctan", "tanh", None]. Default: ``None``.
    adj_norm_func : func of utils.normalize, optional
        Function that normalizes adjacency matrix. Default: ``SAGEAdjNorm``.
    mu : float, optional
        Hyper-parameter, refer to original paper. Default: ``2.0``.
    dropout : float, optional
            Rate of dropout. Default: ``0.0``.

    """

    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 activation=F.relu,
                 layer_norm=False,
                 feat_norm=None,
                 adj_norm_func=SAGEAdjNorm,
                 mu=2.0,
                 dropout=0.0):
        super(GraphSAGE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func

        if type(hidden_features) is int:
            hidden_features = [hidden_features]

        self.layers = nn.ModuleList()
        if layer_norm:
            self.layers.append(nn.LayerNorm(in_features))
        self.layers.append(SAGEConv(in_features=in_features,
                                    pool_features=in_features,
                                    out_features=hidden_features[0],
                                    activation=activation,
                                    mu=mu,
                                    dropout=dropout))
        for i in range(len(hidden_features) - 1):
            if layer_norm:
                self.layers.append(nn.LayerNorm(hidden_features[i]))
            self.layers.append(SAGEConv(in_features=hidden_features[i],
                                        pool_features=hidden_features[i],
                                        out_features=hidden_features[i + 1],
                                        activation=activation,
                                        mu=mu,
                                        dropout=dropout))
        self.layers.append(SAGEConv(in_features=hidden_features[-1],
                                    pool_features=hidden_features[-1],
                                    out_features=out_features,
                                    activation=None,
                                    mu=mu,
                                    dropout=0.0))

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

        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = F.normalize(x, dim=1)
                x = layer(x, adj)

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
    dropout : float, optional
            Rate of dropout. Default: ``0.0``.
    mu : float, optional
        Hyper-parameter, refer to original paper. Default: ``2.0``.

    """

    def __init__(self,
                 in_features,
                 pool_features,
                 out_features,
                 activation=None,
                 mu=2.0,
                 dropout=0.0):
        super(SAGEConv, self).__init__()
        self.pool_layer = nn.Linear(in_features, pool_features)
        self.linear1 = nn.Linear(pool_features, out_features)
        self.linear2 = nn.Linear(pool_features, out_features)
        self.activation = activation
        self.mu = mu

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
        nn.init.xavier_normal_(self.pool_layer.weight, gain=gain)

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

        x = F.relu(self.pool_layer(x))
        x_ = x ** self.mu
        x_ = torch.spmm(adj, x_) ** (1 / self.mu)

        # In original model this is actually max-pool, but **10/**0.1 result in gradient explosion.
        # However we can still achieve similar performance using 2-norm.
        x = self.linear1(x)
        x_ = self.linear2(x_)
        x = x + x_
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)

        return x
