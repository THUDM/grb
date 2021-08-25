import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F

import grb.utils as utils
from grb.model.torch.gcn import GCNConv
from grb.utils.normalize import GCNAdjNorm


class GCNSVD(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 n_layers,
                 activation=F.relu,
                 layer_norm=False,
                 feat_norm=None,
                 adj_norm_func=None,
                 residual=False,
                 dropout=0.0,
                 k=50):
        super(GCNSVD, self).__init__()
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
            self.layers.append(GCNConv(in_features=n_features[i],
                                       out_features=n_features[i + 1],
                                       activation=activation if i != n_layers - 1 else None,
                                       residual=residual if i != n_layers - 1 else False,
                                       dropout=dropout if i != n_layers - 1 else 0.0))
        self.k = k

    @property
    def model_type(self):
        return "torch"

    def forward(self, x, adj):
        adj = self.truncatedSVD(adj, self.k)
        adj = utils.adj_preprocess(adj=adj, adj_norm_func=self.adj_norm_func, device=x.device)
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = layer(x, adj)

        return x

    def truncatedSVD(self, adj, k=50):
        edge_index = adj._indices()
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]

        adj = sp.csr_matrix((np.ones(len(row)), (row, col)))
        if sp.issparse(adj):
            adj = adj.asfptype()
            U, S, V = sp.linalg.svds(adj, k=k)
            diag_S = np.diag(S)
        else:
            U, S, V = np.linalg.svd(adj)
            U = U[:, :k]
            S = S[:k]
            V = V[:k, :]
            diag_S = np.diag(S)

        new_adj = U @ diag_S @ V
        new_adj = sp.csr_matrix(new_adj)

        return new_adj
