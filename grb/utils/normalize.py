import numpy as np
import scipy.sparse as sp


def GCNAdjNorm(adj, order=-0.5):
    adj = sp.eye(adj.shape[0]) + adj
    # for i in range(len(adj.data)):
    #     if adj.data[i] > 0 and adj.data[i] != 1:
    #         adj.data[i] = 1
    adj.data[np.where((adj.data > 0) * (adj.data == 1))[0]] = 1
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, order).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    return adj


def SAGEAdjNorm(adj, order=-1):
    adj = sp.eye(adj.shape[0]) + adj
    for i in range(len(adj.data)):
        if adj.data[i] > 0 and adj.data[i] != 1:
            adj.data[i] = 1
        if adj.data[i] < 0:
            adj.data[i] = 0
    adj.eliminate_zeros()
    adj = sp.coo_matrix(adj)
    if order == 0:
        return adj.tocoo()
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, order).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = d_mat_inv_sqrt @ adj

    return adj


def RobustGCNAdjNorm(adj):
    adj0 = GCNAdjNorm(adj, order=-0.5)
    adj1 = GCNAdjNorm(adj, order=-1)

    return adj0, adj1
