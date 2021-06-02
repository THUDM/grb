import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from grb.model.torch.gcn import GCNConv


class GCNGuard(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, activation=F.relu,
                 layer_norm=False, dropout=True, drop=False, attention=True):
        super(GCNGuard, self).__init__()
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
        self.reset_parameters()
        self.drop = drop
        self.drop_learn = torch.nn.Linear(2, 1)
        self.attention = attention

    @property
    def model_type(self):
        return "torch"

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj, dropout=0):
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                if self.attention:
                    adj = self.att_coef(x, adj)
                x = layer(x, adj, dropout=dropout)

        return x

    def att_coef(self, features, adj):
        edge_index = adj._indices()

        n_node = features.shape[0]
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]

        features_copy = features.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=features_copy, Y=features_copy)  # try cosine similarity
        sim = sim_matrix[row, col]
        sim[sim < 0.1] = 0

        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')

        """add learnable dropout, make character vector"""
        if self.drop:
            character = np.vstack((att_dense_norm[row, col].A1,
                                   att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T).to(features.device)
            drop_score = self.drop_learn(character)
            drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
            mm = torch.nn.Threshold(0.5, 0)
            drop_score = mm(drop_score)
            mm_2 = torch.nn.Threshold(-0.49, 1)
            drop_score = mm_2(-drop_score)
            drop_decision = drop_score.clone().requires_grad_()
            drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
            drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges

        if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
            degree = (att_dense_norm != 0).sum(1).A1
            lam = 1 / (degree + 1)  # degree +1 is to add itself
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight  # add the self loop
        else:
            att = att_dense_norm

        row, col = att.nonzero()
        att_adj = np.vstack((row, col))
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)  # exponent, kind of softmax
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32).to(features.device)
        att_adj = torch.tensor(att_adj, dtype=torch.int64).to(features.device)

        shape = (n_node, n_node)
        new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape)

        return new_adj


class GATGuard(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 num_heads,
                 activation=F.leaky_relu,
                 layer_norm=False,
                 drop=False,
                 attention=True):

        super(GATGuard, self).__init__()
        self.layers = nn.ModuleList()
        if layer_norm:
            self.layers.append(nn.LayerNorm(in_features))
        self.layers.append(
            GATConv(in_features, hidden_features[0], num_heads, activation=activation, allow_zero_in_degree=True))
        for i in range(len(hidden_features) - 1):
            if layer_norm:
                self.layers.append(nn.LayerNorm(hidden_features[i] * num_heads))
            self.layers.append(
                GATConv(hidden_features[i] * num_heads, hidden_features[i + 1], num_heads, activation=activation,
                        allow_zero_in_degree=True))
        self.layers.append(GATConv(hidden_features[-1] * num_heads, num_heads, out_features, allow_zero_in_degree=True))
        self.drop = drop
        self.drop_learn = torch.nn.Linear(2, 1)
        self.attention = attention

    @property
    def model_type(self):
        return "dgl"

    def forward(self, x, adj, dropout=0):
        graph = dgl.from_scipy(adj).to(x.device)
        graph.ndata['features'] = x

        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                if self.attention:
                    adj = self.att_coef(x, adj)
                    graph = dgl.from_scipy(adj).to(x.device)
                    graph.ndata['features'] = x
                x = layer(graph, x).flatten(1)
                if i != len(self.layers) - 1:
                    x = F.dropout(x, dropout)

        return x

    def att_coef(self, features, adj):
        adj = adj.tocoo()
        n_node = features.shape[0]
        row, col = adj.row, adj.col

        features_copy = features.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=features_copy, Y=features_copy)  # try cosine similarity
        sim = sim_matrix[row, col]
        sim[sim < 0.1] = 0

        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')

        """add learnable dropout, make character vector"""
        if self.drop:
            character = np.vstack((att_dense_norm[row, col].A1,
                                   att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T).to(features.device)
            drop_score = self.drop_learn(character)
            drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
            mm = torch.nn.Threshold(0.5, 0)
            drop_score = mm(drop_score)
            mm_2 = torch.nn.Threshold(-0.49, 1)
            drop_score = mm_2(-drop_score)
            drop_decision = drop_score.clone().requires_grad_()
            drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
            drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges

        if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
            degree = (att_dense_norm != 0).sum(1).A1
            lam = 1 / (degree + 1)  # degree +1 is to add itself
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight  # add the self loop
        else:
            att = att_dense_norm

        row, col = att.nonzero()
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)
        att_edge_weight = np.asarray(att_edge_weight.ravel())[0]
        new_adj = sp.csr_matrix((att_edge_weight, (row, col)))

        return new_adj
