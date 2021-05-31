import dgl
import numpy as np
import torch as th
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F


def drop_node(feats, drop_rate, training):
    n = feats.shape[0]
    drop_rates = th.FloatTensor(np.ones(n) * drop_rate)

    if training:

        masks = th.bernoulli(1. - drop_rates).unsqueeze(1)
        feats = masks.to(feats.device) * feats

    else:
        feats = feats * (1. - drop_rate)

    return feats


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn=False):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.input_dropout = nn.Dropout(input_droprate)
        self.hidden_dropout = nn.Dropout(hidden_droprate)
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, x):

        if self.use_bn:
            x = self.bn1(x)
        x = self.input_dropout(x)
        x = F.relu(self.layer1(x))

        if self.use_bn:
            x = self.bn2(x)
        x = self.hidden_dropout(x)
        x = self.layer2(x)

        return x


def GRANDConv(graph, feats, order):
    '''
    Parameters
    -----------
    graph: dgl.Graph
        The input graph
    feats: Tensor (n_nodes * feat_dim)
        Node features
    order: int
        Propagation Steps
    '''
    with graph.local_scope():
        ''' Calculate Symmetric normalized adjacency matrix   \hat{A} '''
        degs = graph.in_degrees().float().clamp(min=1)
        norm = th.pow(degs, -0.5).to(feats.device).unsqueeze(1)

        graph.ndata['norm'] = norm
        graph.apply_edges(fn.u_mul_v('norm', 'norm', 'weight'))

        ''' Graph Conv '''
        x = feats
        y = 0 + feats

        for i in range(order):
            graph.ndata['h'] = x
            graph.update_all(fn.u_mul_e('h', 'weight', 'm'), fn.sum('m', 'h'))
            x = graph.ndata.pop('h')
            y.add_(x)

    return y / (order + 1)


class GRAND(nn.Module):
    r"""
    Parameters
    -----------
    in_features: int
        Input feature size. i.e, the number of dimensions of: math: `H^{(i)}`.
    hidden_features: int
        Hidden feature size.
    out_features: int
        Number of classes.
    S: int
        Number of Augmentation samples
    K: int
        Number of Propagation Steps
    node_dropout: float
        Dropout rate on node features.
    input_dropout: float
        Dropout rate of the input layer of a MLP
    hidden_dropout: float
        Dropout rate of the hidden layer of a MLPx
    batchnorm: bool, optional
        If True, use batch normalization.
    """

    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 S=1,
                 K=3,
                 node_dropout=0.0,
                 input_droprate=0.0,
                 hidden_droprate=0.0,
                 batchnorm=False):

        super(GRAND, self).__init__()
        self.in_dim = in_features
        self.hid_dim = hidden_features
        self.S = S
        self.K = K
        self.n_class = out_features

        self.mlp = MLP(in_features, hidden_features, out_features, input_droprate, hidden_droprate, batchnorm)

        self.dropout = node_dropout
        self.node_dropout = nn.Dropout(node_dropout)

    def forward(self, x, adj, dropout=0, training=True):
        graph = dgl.from_scipy(adj).to(x.device)
        graph.ndata['features'] = x

        S = self.S

        if training:  # Training Mode
            output_list = []
            for s in range(S):
                drop_feat = drop_node(x, self.dropout, True)  # Drop node
                feat = GRANDConv(graph, drop_feat, self.K)  # Graph Convolution
                output_list.append(th.log_softmax(self.mlp(feat), dim=-1))  # Prediction

            return output_list
        else:  # Inference Mode
            drop_feat = drop_node(x, self.dropout, False)
            x = GRANDConv(graph, drop_feat, self.K)
            x = self.mlp(x)

            return x

    @property
    def model_type(self):
        return "dgl"
