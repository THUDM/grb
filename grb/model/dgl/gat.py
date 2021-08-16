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
                 num_heads,
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
            hidden_features = [hidden_features]

        self.layers = nn.ModuleList()
        if layer_norm:
            self.layers.append(nn.LayerNorm(in_features))
        self.layers.append(GATConv(in_feats=in_features,
                                   out_feats=hidden_features[0],
                                   num_heads=num_heads,
                                   feat_drop=feat_dropout,
                                   attn_drop=attn_dropout,
                                   residual=residual,
                                   activation=activation))
        for i in range(len(hidden_features) - 1):
            if layer_norm:
                self.layers.append(nn.LayerNorm(hidden_features[i] * num_heads))
            self.layers.append(GATConv(in_feats=hidden_features[i] * num_heads,
                                       out_feats=hidden_features[i + 1],
                                       num_heads=num_heads,
                                       feat_drop=feat_dropout,
                                       attn_drop=attn_dropout,
                                       residual=residual,
                                       activation=activation))
        self.layers.append(GATConv(in_feats=hidden_features[-1] * num_heads,
                                   out_feats=out_features,
                                   num_heads=num_heads,
                                   feat_drop=0.0,
                                   attn_drop=0.0,
                                   residual=False,
                                   activation=None))
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
