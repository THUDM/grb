import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv

from grb.utils.normalize import GCNAdjNorm


class GAT(nn.Module):
    r"""

    Description
    -----------
    Graph Attention Networks (`GAT <https://arxiv.org/abs/1710.10903>`__)

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
        Activation function. Default: ``torch.nn.functional.leaky_relu``.
    feat_norm : str, optional
        Type of features normalization, choose from ["arctan", "tanh", None]. Default: ``None``.
    adj_norm_func : func of utils.normalize, optional
        Function that normalizes adjacency matrix. Default: ``GCNAdjNorm``.
    feat_dropout : float, optional
        Dropout rate for input features. Default: ``0.0``.
    attn_dropout : float, optional
        Dropout rate for attention. Default: ``0.0``.
    residual : bool, optional
        Whether to use residual connection. Default: ``False``.
    dropout : float, optional
        Dropout rate during training. Default: ``0.0``.

    """
    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 n_layers,
                 n_heads,
                 activation=F.leaky_relu,
                 layer_norm=False,
                 feat_norm=None,
                 adj_norm_func=GCNAdjNorm,
                 feat_dropout=0.0,
                 attn_dropout=0.0,
                 residual=False,
                 dropout=0.0):
        super(GAT, self).__init__()
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
            self.layers.append(GATConv(in_feats=n_features[i] * n_heads if i != 0 else n_features[i],
                                       out_feats=n_features[i + 1],
                                       num_heads=n_heads if i != n_layers - 1 else 1,
                                       feat_drop=feat_dropout if i != n_layers - 1 else 0.0,
                                       attn_drop=attn_dropout if i != n_layers - 1 else 0.0,
                                       residual=residual if i != n_layers - 1 else False,
                                       activation=activation if i != n_layers - 1 else None))
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    @property
    def model_type(self):
        return "dgl"

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

        graph = dgl.from_scipy(adj).to(x.device)
        graph = dgl.add_self_loop(graph)
        graph.ndata['features'] = x

        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = layer(graph, x).flatten(1)
                if i != len(self.layers) - 1:
                    if self.dropout is not None:
                        x = self.dropout(x)

        return x
