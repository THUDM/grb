import argparse
import os
import pickle

import models
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn.functional as F

weight = [0.3, 0.24, 0.18, 0.12, 0.08, 0.05, 0.03]


def adj_to_tensor(adj):
    sparse_row = torch.LongTensor(adj.row).unsqueeze(1)
    sparse_col = torch.LongTensor(adj.col).unsqueeze(1)
    sparse_concat = torch.cat((sparse_row, sparse_col), 1)
    sparse_data = torch.FloatTensor(adj.data)
    adj_tensor = torch.sparse.FloatTensor(sparse_concat.t(), sparse_data, torch.Size(adj.shape))

    return adj_tensor


class CustomDataset(object):
    def __init__(self, adj, features, labels, train_mask, val_mask, test_mask, name=None, mode='normal', verbose=True):
        self.adj = adj.tocoo()
        self.adj_tensor = adj_to_tensor(self.adj)
        self.num_nodes = features.shape[0]
        self.num_edges = adj.getnnz()
        self.num_features = features.shape[1]

        if type(features) != torch.Tensor:
            features = torch.FloatTensor(features)
        elif features.type() != 'torch.FloatTensor':
            features = features.float()
        self.features = features

        if type(labels) != torch.Tensor:
            labels = torch.LongTensor(labels)
        elif labels.type() != 'torch.LongTensor':
            labels = labels.long()
        self.labels = labels

        if type(train_mask) != torch.Tensor:
            train_mask = torch.BoolTensor(train_mask)
        elif train_mask.type() != 'torch.BoolTensor':
            train_mask = train_mask.bool()
        self.train_mask = train_mask

        if type(val_mask) != torch.Tensor:
            val_mask = torch.BoolTensor(val_mask)
        elif val_mask.type() != 'torch.BoolTensor':
            val_mask = val_mask.bool()
        self.val_mask = val_mask

        if type(test_mask) != torch.Tensor:
            test_mask = torch.BoolTensor(test_mask)
        elif test_mask.type() != 'torch.BoolTensor':
            test_mask = test_mask.bool()
        self.test_mask = test_mask

        if mode == 'lcc':
            graph = nx.from_scipy_sparse_matrix(adj)
            components = nx.connected_components(graph)
            lcc_nodes = list(next(components))
            subgraph = graph.subgraph(lcc_nodes)
            self.adj = nx.to_scipy_sparse_matrix(subgraph, format='coo')
            self.adj_tensor = adj_to_tensor(self.adj)
            self.features = self.features[lcc_nodes]
            self.labels = self.labels[lcc_nodes]
            self.train_mask = self.train_mask[lcc_nodes]
            self.val_mask = self.val_mask[lcc_nodes]
            self.test_mask = self.test_mask[lcc_nodes]
            self.num_train = int(torch.sum(self.train_mask))
            self.num_val = int(torch.sum(self.val_mask))
            self.num_test = int(torch.sum(self.test_mask))
            self.num_nodes = subgraph.number_of_nodes()
            self.num_edges = subgraph.number_of_edges() // 2

        self.num_train = int(torch.sum(self.train_mask))
        self.num_val = int(torch.sum(self.val_mask))
        self.num_test = int(torch.sum(self.test_mask))
        self.num_classes = int(labels.max() + 1)

        if verbose:
            print("Custom Dataset \'{}\' loaded.".format(name))
            print("    Number of nodes: {}.".format(self.num_nodes))
            print("    Number of edges: {}.".format(self.num_edges))
            print("    Number of features: {}.".format(self.num_features))
            print("    Number of classes: {}.".format(self.num_classes))
            print("    Number of train samples: {}.".format(self.num_train))
            print("    Number of val samples: {}.".format(self.num_val))
            print("    Number of test samples: {}.".format(self.num_test))
            print("    Feature range [{:.4f}, {:.4f}]".format(self.features.min(), self.features.max()))


def eval_acc(pred, labels, mask=None):
    if mask is None:
        return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)
    else:
        if torch.sum(mask) != 0:
            return (torch.argmax(pred[mask], dim=1) == labels[mask]).float().sum() / int(torch.sum(mask))
        else:
            return 0.0


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

    return adj.tocoo()


def RobustGCNAdjNorm(adj):
    adj0 = GCNAdjNorm(adj, order=-0.5)
    adj1 = GCNAdjNorm(adj, order=-1)

    return adj0, adj1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GIA Evaluation')
    parser.add_argument("--gpu", type=int, default=1, help="gpu")
    parser.add_argument("--n_epoch", type=int, default=0, help="Training epoch.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--data_dir", type=str, default="/data/qinkai/aminer_revised/")
    parser.add_argument("--model_list", nargs='+', default=['gcn_ln', 'graphsage', 'sgcn',
                                                            'robustgcn', 'tagcn', 'appnp', 'gin'])
    parser.add_argument("--model_dir", type=str, default="./eval_models/")
    parser.add_argument("--model_suffix", type=str, default="aminer")
    parser.add_argument("--attack_dir", type=str, default="./submission/")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.gpu >= 0:
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    # Data loading
    data_dir = args.data_dir
    with open(os.path.join(data_dir, "adj.pkl"), 'rb') as f:
        adj = pickle.load(f)
    labels = np.load(os.path.join(data_dir, "labels_train.npy"))
    features = np.load(os.path.join(data_dir, "features.npy"))
    labels_test = np.load(os.path.join(data_dir, "labels_test.npy"))
    labels = np.concatenate([labels, labels_test], axis=0)

    n_node = features.shape[0]
    n_val = 50000  # user-defined val size
    n_test = 50000

    train_mask = torch.zeros(n_node, dtype=bool)
    train_mask[range(n_node - n_val - n_test)] = True
    val_mask = torch.zeros(n_node, dtype=bool)
    val_mask[range(n_node - n_val - n_test, n_node - n_test)] = True
    test_mask = torch.zeros(n_node, dtype=bool)
    test_mask[range(n_node - n_test, n_node)] = True

    if args.attack_dir != '':
        features_attack = np.load(os.path.join(args.attack_dir, "features.npy"))
        with open(os.path.join(args.attack_dir, "adj.pkl"), 'rb') as f:
            adj_attack = pickle.load(f)
        adj_attack = sp.csr_matrix(adj_attack)
        adj_attacked = sp.vstack([adj, adj_attack[:, :n_node]])
        adj_attacked = sp.hstack([adj_attacked, adj_attack.T])
        adj_attacked = sp.csr_matrix(adj_attacked)
        features_attacked = np.concatenate([features, features_attack])

        dataset = CustomDataset(adj=adj_attacked,
                                features=features_attacked,
                                labels=labels,
                                train_mask=train_mask,
                                val_mask=val_mask,
                                test_mask=test_mask,
                                name='Aminer')

    else:
        dataset = CustomDataset(adj=adj,
                                features=features,
                                labels=labels,
                                train_mask=train_mask,
                                val_mask=val_mask,
                                test_mask=test_mask,
                                name='Aminer')

    # Data preprocessing
    adj = dataset.adj
    adj_tensor = dataset.adj_tensor
    features = dataset.features
    labels = dataset.labels
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    features = torch.FloatTensor(features).to(device)
    features[torch.where(features < -0.4)[0]] = 0
    features[torch.where(features > 0.4)[0]] = 0
    features[np.where(adj.getnnz(axis=1) > 90)[0]] = 0
    labels = torch.LongTensor(labels).to(device)

    test_acc_dict = {}
    for model_name in args.model_list:
        if model_name in "gcn_ln":
            model = models.GCN(in_features=num_features, out_features=num_classes,
                               hidden_features=[256, 128, 64], layer_norm=True, activation=F.relu)

        elif model_name in "graphsage":
            model = models.GraphSAGE(in_features=num_features, out_features=num_classes,
                                     hidden_features=[128, 128, 128], activation=F.relu)

        elif model_name in "sgcn":
            model = models.SGCN(in_features=num_features, out_features=num_classes,
                                hidden_features=[128, 128, 128],
                                activation=F.relu)

        elif model_name in "robustgcn":
            model = models.RobustGCN(in_features=num_features, out_features=num_classes,
                                     hidden_features=[128, 128, 128])

        elif model_name in "tagcn":
            model = models.TAGCN(in_features=num_features, out_features=num_classes,
                                 hidden_features=[128, 128, 128],
                                 k=2, activation=F.leaky_relu)

        elif model_name in "appnp":
            model = models.APPNP(in_features=num_features, out_features=num_classes,
                                 hidden_features=128, alpha=0.01, k=10)

        elif model_name in "gin":
            model = models.GIN(in_features=num_features, out_features=num_classes,
                               hidden_features=[128, 128, 128],
                               activation=F.relu)

        model.load_state_dict(torch.load(os.path.join(args.model_dir, model_name + "_" + args.model_suffix + ".pt")))
        model.to(device)
        model.eval()
        if model_name in "robustgcn":
            adj_ = RobustGCNAdjNorm(adj)
        elif model_name in "graphsage":
            adj_ = SAGEAdjNorm(adj)
        else:
            adj_ = GCNAdjNorm(adj)

        if type(adj_) is tuple:
            adj_ = [adj_to_tensor(adj).to(device) for adj in adj_]
        else:
            adj_ = adj_to_tensor(adj_).to(device)

        logits = model(features, adj_, dropout=0)
        logp = F.softmax(logits[:n_node], 1)
        test_acc = eval_acc(logp, labels, test_mask)
        test_acc_dict[model_name] = test_acc.cpu().numpy()
        print("Test score of {}: {:.4f}".format(model_name, test_acc))

    # print("Test ACC dict:", test_acc_dict)
    test_acc_sorted = sorted(list(test_acc_dict.values()))
    final_score = 0.0
    for i in range(len(weight)):
        final_score += weight[i] * test_acc_sorted[i]

    print("Average score: {:.4f}".format(np.average(test_acc_sorted)))
    print("3-max score: {:.4f}".format(np.average(test_acc_sorted[-3:])))
    print("Weighted score: {:.4f}".format(final_score))
