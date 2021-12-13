import torch
import numpy as np
import scipy.sparse as sp


def GCNAdjNorm(adj, order=-0.5):
    r"""

    Description
    -----------
    Normalization of adjacency matrix proposed in `GCN <https://arxiv.org/abs/1609.02907>`__.

    Parameters
    ----------
    adj : scipy.sparse.csr.csr_matrix or torch.FloatTensor
        Adjacency matrix in form of ``N * N`` sparse matrix (or in form of ``N * N`` dense tensor).
    order : float, optional
        Order of degree matrix. Default: ``-0.5``.


    Returns
    -------
    adj : scipy.sparse.csr.csr_matrix
        Normalized adjacency matrix in form of ``N * N`` sparse matrix.

    """
    if sp.issparse(adj):
        adj = sp.eye(adj.shape[0]) + adj
        adj.data[np.where((adj.data > 0) * (adj.data == 1))[0]] = 1
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, order).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj = d_mat_inv @ adj @ d_mat_inv
    else:
        rowsum = torch.sparse.mm(adj, torch.ones((adj.shape[0], 1), device=adj.device)) + 1
        d_inv = torch.pow(rowsum, order).flatten()
        d_inv[torch.isinf(d_inv)] = 0.

        self_loop_idx = torch.stack((
            torch.arange(adj.shape[0], device=adj.device),
            torch.arange(adj.shape[0], device=adj.device)
        ))
        self_loop_val = torch.ones_like(self_loop_idx[0], dtype=adj.dtype)
        indices = torch.cat((self_loop_idx, adj.indices()), dim=1)
        values = torch.cat((self_loop_val, adj.values()))
        values = d_inv[indices[0]] * values * d_inv[indices[1]]
        adj = torch.sparse.FloatTensor(indices, values, adj.shape).coalesce()

    return adj


def SAGEAdjNorm(adj, order=-1):
    r"""

    Description
    -----------
    Normalization of adjacency matrix proposed in `GraphSAGE <https://arxiv.org/abs/1706.02216>`__.

    Parameters
    ----------
    adj : scipy.sparse.csr.csr_matrix
        Adjacency matrix in form of ``N * N`` sparse matrix.
    order : float, optional
        Order of degree matrix. Default: ``-0.5``.


    Returns
    -------
    adj : scipy.sparse.csr.csr_matrix
        Normalized adjacency matrix in form of ``N * N`` sparse matrix.

    """
    if sp.issparse(adj):
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
        d_inv = np.power(rowsum, order).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj = d_mat_inv @ adj
    else:
        adj = torch.eye(adj.shape[0]).to(adj.device) + adj
        rowsum = adj.sum(1)
        d_inv = torch.pow(rowsum, order).flatten()
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diag(d_inv)
        adj = d_mat_inv @ adj

    return adj


def SPARSEAdjNorm(adj, order=-0.5):
    r"""

    Description
    -----------
    Normalization of adjacency matrix proposed in `GCN <https://arxiv.org/abs/1609.02907>`__.

    Parameters
    ----------
    adj : scipy.sparse.csr.csr_matrix or torch.FloatTensor
        Adjacency matrix in form of ``N * N`` sparse matrix (or in form of ``N * N`` dense tensor).
    order : float, optional
        Order of degree matrix. Default: ``-0.5``.


    Returns
    -------
    adj : scipy.sparse.csr.csr_matrix
        Normalized adjacency matrix in form of ``N * N`` sparse matrix.

    """
    if sp.issparse(adj):
        adj = sp.eye(adj.shape[0]) + adj
        adj.data[np.where((adj.data > 0) * (adj.data == 1))[0]] = 1
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, order).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj = d_mat_inv @ adj @ d_mat_inv
    else:
        rowsum = torch.sparse.mm(adj, torch.ones((adj.shape[0], 1), device=adj.device)) + 1
        d_inv = torch.pow(rowsum, order).flatten()
        d_inv[torch.isinf(d_inv)] = 0.

        self_loop_idx = torch.stack((
            torch.arange(adj.shape[0], device=adj.device),
            torch.arange(adj.shape[0], device=adj.device)
        ))
        self_loop_val = torch.ones_like(self_loop_idx[0], dtype=adj.dtype)
        indices = torch.cat((self_loop_idx, adj.indices()), dim=1)
        values = torch.cat((self_loop_val, adj.values()))
        values = d_inv[indices[0]] * values * d_inv[indices[1]]
        adj = torch.sparse.FloatTensor(indices, values, adj.shape).coalesce()

    return adj


def RobustGCNAdjNorm(adj):
    r"""

    Description
    -----------
    Normalization of adjacency matrix proposed in `RobustGCN <http://pengcui.thumedialab.com/papers/RGCN.pdf>`__.

    Parameters
    ----------
    adj : tuple of scipy.sparse.csr.csr_matrix
        Tuple of adjacency matrix in form of ``N * N`` sparse matrix.

    Returns
    -------
    adj0 : scipy.sparse.csr.csr_matrix
        Adjacency matrix in form of ``N * N`` sparse matrix.
    adj1 : scipy.sparse.csr.csr_matrix
        Adjacency matrix in form of ``N * N`` sparse matrix.

    """
    adj0 = GCNAdjNorm(adj, order=-0.5)
    adj1 = GCNAdjNorm(adj, order=-1)

    return adj0, adj1


def feature_normalize(features):
    x_sum = torch.sum(features, dim=1)
    x_rev = x_sum.pow(-1).flatten()
    x_rev[torch.isnan(x_rev)] = 0.0
    x_rev[torch.isinf(x_rev)] = 0.0
    features = features * x_rev.unsqueeze(-1).expand_as(features)

    return features
