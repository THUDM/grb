import argparse
import os
import pickle

import models
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def GCNAdjNorm(adj, order=-0.5):
    adj = sp.eye(adj.shape[0]) + adj
    for i in range(len(adj.data)):
        if adj.data[i] > 0 and adj.data[i] != 1:
            adj.data[i] = 1
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, order).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    return adj.tocoo()


def adj_to_tensor(adj):
    sparse_row = torch.LongTensor(adj.row).unsqueeze(1)
    sparse_col = torch.LongTensor(adj.col).unsqueeze(1)
    sparse_concat = torch.cat((sparse_row, sparse_col), 1)
    sparse_data = torch.FloatTensor(adj.data)
    adj_tensor = torch.sparse.FloatTensor(sparse_concat.t(), sparse_data, torch.Size(adj.shape))

    return adj_tensor


def run_model(adj_sets, features_sets, device):
    model = models.TAGCN(in_features=100, out_features=18,
                         hidden_features=[128, 128, 128],
                         k=2, activation=F.leaky_relu)

    model.load_state_dict(torch.load("eval_models/tagcn_aminer.pt"))
    model.to(device)
    model.eval()
    answers = []
    with torch.no_grad():
        for i in range(len(adj_sets)):
            print("running data ", i)
            features = features_sets[i]
            features = torch.FloatTensor(features).to(device)
            adj = adj_sets[i]

            features[torch.where(features < -0.4)[0]] = 0
            features[torch.where(features > 0.4)[0]] = 0
            features[np.where(adj.getnnz(axis=1) > 90)[0]] = 0

            adj_ = GCNAdjNorm(adj)
            print(np.min(adj_.col))
            print(np.min(adj_.row))
            if type(adj_) is tuple:
                adj_ = [adj_to_tensor(adj).to(device) for adj in adj_]
            else:
                adj_ = adj_to_tensor(adj_).to(device)
            print(adj_)
            logits = model(features, adj_, dropout=0)
            answer = torch.argmax(logits, -1).data.cpu().numpy()
            answers.append(answer)
            del adj
            del adj_
            del features
            torch.cuda.empty_cache()

        return answers
