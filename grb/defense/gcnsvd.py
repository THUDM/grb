import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

import grb.utils as utils
from grb.model.torch.gcn import GCNConv


class GCNSVD(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, activation=F.relu,
                 layer_norm=False, dropout=True, k=50):
        super(GCNSVD, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if type(hidden_features) is int:
            hidden_features = [hidden_features]

        self.layers = nn.ModuleList()
        if layer_norm:
            self.layers.append(nn.LayerNorm(in_features))
        self.layers.append(GCNConv(in_features, hidden_features[0], activation=activation, dropout=dropout))
        for i in range(len(hidden_features) - 1):
            if layer_norm:
                self.layers.append(nn.LayerNorm(hidden_features[i]))
            self.layers.append(
                GCNConv(hidden_features[i], hidden_features[i + 1], activation=activation, dropout=dropout))
        self.layers.append(GCNConv(hidden_features[-1], out_features))
        self.k = k

    @property
    def model_type(self):
        return "torch"

    def forward(self, x, adj, dropout=0):
        adj = self.truncatedSVD(adj, self.k)
        adj = utils.adj_preprocess(adj=adj, device=x.device)
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = layer(x, adj, dropout=dropout)

        return x

    def truncatedSVD(self, adj, k=50, verbose=False):
        edge_index = adj._indices()
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]

        adj = sp.csr_matrix((np.ones(len(row)), (row, col)))
        # print('=== GCN-SVD: rank={} ==='.format(k))
        if sp.issparse(adj):
            adj = adj.asfptype()
            U, S, V = sp.linalg.svds(adj, k=k)
            # print("rank_after = {}".format(len(S.nonzero()[0])))
            diag_S = np.diag(S)
        else:
            U, S, V = np.linalg.svd(adj)
            U = U[:, :k]
            S = S[:k]
            V = V[:k, :]
            # print("rank_before = {}".format(len(S.nonzero()[0])))
            diag_S = np.diag(S)
            # print("rank_after = {}".format(len(diag_S.nonzero()[0])))

        new_adj = U @ diag_S @ V
        new_adj = sp.csr_matrix(new_adj)

        return new_adj
