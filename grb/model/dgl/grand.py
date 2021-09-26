import dgl
import torch
import numpy as np
import torch.nn as nn
import dgl.function as fn

from grb.model.torch import MLP


class GRAND(nn.Module):
    r"""

    Description
    -----------
    Graph Random Neural Networks (`GRAND <https://arxiv.org/abs/2005.11079>`__)


    Parameters
    -----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    hidden_features : int or list of int
        Dimension of hidden features. List if multi-layer.
    n_layers : int
        Number of layers.
    s: int
        Number of Augmentation samples
    k: int
        Number of Propagation Steps
    node_dropout: float
        Dropout rate on node features.
    input_dropout: float
        Dropout rate of the input layer of a MLP
    hidden_dropout: float
        Dropout rate of the hidden layer of a MLP
    """

    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 n_layers=2,
                 s=1,
                 k=3,
                 temp=1.0,
                 lam=1.0,
                 feat_norm=None,
                 adj_norm_func=None,
                 node_dropout=0.0,
                 input_dropout=0.0,
                 hidden_dropout=0.0):
        super(GRAND, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        self.s = s
        self.k = k
        self.temp = temp
        self.lam = lam
        self.mlp = MLP(in_features=in_features,
                       out_features=out_features,
                       hidden_features=hidden_features,
                       n_layers=n_layers,
                       dropout=hidden_dropout)
        self.node_dropout = node_dropout
        if input_dropout > 0.0:
            self.input_dropout = nn.Dropout(input_dropout)
        else:
            self.input_dropout = None

    def forward(self, x, adj):
        graph = dgl.from_scipy(adj).to(x.device)
        graph.ndata['features'] = x

        if self.input_dropout is not None:
            x = self.input_dropout(x)
        if self.training:
            output_list = []
            for s in range(self.s):
                x_drop = drop_node(x, self.node_dropout, training=True)
                x_drop = GRANDConv(graph, x_drop, hop=self.k)
                output_list.append(torch.log_softmax(self.mlp(x_drop), dim=-1))

            return output_list
        else:
            x = GRANDConv(graph, x, self.k)
            x = self.mlp(x)

            return x

    @property
    def model_type(self):
        return "dgl"

    @property
    def model_name(self):
        return "grand"


def drop_node(features, drop_rate, training=True):
    n = features.shape[0]
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)

    if training:
        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
        features = masks.to(features.device) * features

    return features


def GRANDConv(graph, features, hop):
    r"""

    Parameters
    -----------
    graph : dgl.Graph
        The input graph
    features : torch.Tensor
        Tensor of node features
    hop : int
        Propagation Steps

    """

    with graph.local_scope():
        degs = graph.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5).to(features.device).unsqueeze(1)

        graph.ndata['norm'] = norm
        graph.apply_edges(fn.u_mul_v('norm', 'norm', 'weight'))

        x = features
        y = 0.0 + features

        for i in range(hop):
            graph.ndata['h'] = x
            graph.update_all(fn.u_mul_e('h', 'weight', 'm'), fn.sum('m', 'h'))
            x = graph.ndata.pop('h')
            y.add_(x)

    return y / (hop + 1)
