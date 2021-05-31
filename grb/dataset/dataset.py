import os

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from cogdl.datasets import build_dataset_from_name, build_dataset_from_path

from grb.dataset.URLs import URLs
from grb.utils import download


class Dataset(object):
    r"""
    Description
    -----------
    Dataset class that helps to load GRB datasets for evaluating adversarial robustness.

    Parameters
    ----------
    name: str
        Name of the dataset, supported datasets: ["grb-cora", "grb-aminer", "grb-reddit", "grb-amazon"]
    data_dir: str, optional
        Directory for dataset. If not provided, default is "./data/".
    mode: str, optional
        Difficulty of the dataset, which is determined mainly according to the average degree of nodes.
        Choose from ["easy", "medium", "hard", "full"]. "full" is to use the entire test set.
    feat_norm: str, optional
        Feature normalization that transform all features to range [-1, 1].
        Choose from ["arctan", "sigmoid", "tanh"].
    verbose: bool, optional
        Whether to display logs.

    Attributes
    ----------


    Note
    ----
    TBD

    Examples
    --------
    TBD
    """

    def __init__(self, name, data_dir=None, mode="easy", feat_norm=None, verbose=True):
        # Create data dir
        if name not in self.GRB_DATASETS:
            print("{} dataset not supported.".format(name))
            exit(1)
        if data_dir is None:
            data_dir = os.path.join("./data", name)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Load adj
        adj_name = "adj.npz"
        if not os.path.exists(os.path.join(data_dir, adj_name)):
            download(url=URLs[name][adj_name],
                     save_path=os.path.join(data_dir, adj_name))
        adj = sp.load_npz(os.path.join(data_dir, adj_name))

        # Load features
        features_name = "features.npz"
        if not os.path.exists(os.path.join(data_dir, features_name)):
            download(url=URLs[name][features_name],
                     save_path=os.path.join(data_dir, features_name))
        features = np.load(os.path.join(data_dir, features_name)).get("data")
        if feat_norm is not None:
            features = feat_normalize(features, norm=feat_norm)

        # Load labels
        labels_name = "labels.npz"
        if not os.path.exists(os.path.join(data_dir, labels_name)):
            download(url=URLs[name][labels_name],
                     save_path=os.path.join(data_dir, labels_name))
        labels = np.load(os.path.join(data_dir, labels_name)).get("data")

        self.adj = adj
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.num_nodes = features.shape[0]
        self.num_edges = adj.getnnz() // 2
        self.num_features = features.shape[1]
        self.mode = mode
        if len(labels.shape) == 1:
            self.num_classes = int(labels.max() + 1)
        else:
            self.num_classes = labels.shape[-1]

        # Load index
        index_name = "index.npz"
        if not os.path.exists(os.path.join(data_dir, index_name)):
            download(url=URLs[name][index_name],
                     save_path=os.path.join(data_dir, index_name))
        index = np.load(os.path.join(data_dir, index_name))
        index_train = index.get("index_train")
        train_mask = torch.zeros(self.num_nodes, dtype=bool)
        train_mask[index_train] = True
        self.index_train = index_train
        self.train_mask = train_mask

        index_val = index.get("index_val")
        val_mask = torch.zeros(self.num_nodes, dtype=bool)
        val_mask[index_val] = True
        self.index_val = index_val
        self.val_mask = val_mask

        if mode == "easy":
            index_test = index.get("index_test_easy")
        elif mode == "medium":
            index_test = index.get("index_test_medium")
        elif mode == "hard":
            index_test = index.get("index_test_hard")
        elif mode == "full":
            index_test = index.get("index_test")
        else:
            index_test = index.get("index_test")

        test_mask = torch.zeros(self.num_nodes, dtype=bool)
        test_mask[index_test] = True
        self.index_test = index_test
        self.test_mask = test_mask

        self.num_train = int(torch.sum(self.train_mask))
        self.num_val = int(torch.sum(self.val_mask))
        self.num_test = int(torch.sum(self.test_mask))

        if verbose:
            print("Dataset \'{}\' loaded.".format(name))
            print("    Number of nodes: {}".format(self.num_nodes))
            print("    Number of edges: {}".format(self.num_edges))
            print("    Number of features: {}".format(self.num_features))
            print("    Number of classes: {}".format(self.num_classes))
            print("    Number of train samples: {}".format(self.num_train))
            print("    Number of val samples: {}".format(self.num_val))
            print("    Number of test samples: {}".format(self.num_test))
            print("    Dataset mode: {}".format(self.mode))
            print("    Feature range: [{:.4f}, {:.4f}]".format(self.features.min(), self.features.max()))

    GRB_DATASETS = {"grb-cora",
                    "grb-citeseer",
                    "grb-aminer",
                    "grb-reddit",
                    "grb-amazon",
                    "grb-yelp",
                    "grb-ppi",
                    "grb-flickr"}


class CogDLDataset(object):
    def __init__(self, name, data_dir=None, mode='origin', verbose=True):
        """

        :param name:
        :param data_dir: e.g. /data/Cora/;
        :param mode: 'normal', 'lcc';
        :param verbose:
        """

        try:
            if data_dir:
                dataset = build_dataset_from_path(data_path=data_dir, task="node_classification", dataset=name)
            else:
                dataset = build_dataset_from_name(name)
        except AssertionError:
            print("Dataset '{}' is not supported.".format(name))
            exit(1)

        graph = dataset.data
        edge_index = graph.edge_index
        attr = graph.edge_attr if graph.edge_attr is not None else torch.ones(edge_index[0].shape[0])
        self.adj = self.build_adj(attr, edge_index, adj_type='csr')

        if mode == 'origin':
            self.features = dataset.data.x
            self.labels = dataset.data.y
            self.train_mask = dataset.data.train_mask
            self.val_mask = dataset.data.val_mask
            self.test_mask = dataset.data.test_mask
            self.num_train = int(torch.sum(self.train_mask))
            self.num_val = int(torch.sum(self.val_mask))
            self.num_test = int(torch.sum(self.test_mask))
            self.num_nodes = dataset.data.num_nodes
            self.num_edges = dataset.data.num_edges // 2
            self.num_features = dataset.data.num_features
            self.num_classes = dataset.data.num_classes
        elif mode == 'lcc':
            # Get largest connected component
            graph_nx = nx.from_scipy_sparse_matrix(self.adj)
            components = nx.connected_components(graph_nx)
            lcc_nodes = list(next(components))
            subgraph = graph_nx.subgraph(lcc_nodes)
            self.adj = nx.to_scipy_sparse_matrix(subgraph, format='coo')
            self.features = dataset.data.x[lcc_nodes]
            self.labels = dataset.data.y[lcc_nodes]
            self.train_mask = dataset.data.train_mask[lcc_nodes]
            self.val_mask = dataset.data.val_mask[lcc_nodes]
            self.test_mask = dataset.data.test_mask[lcc_nodes]
            self.num_train = int(torch.sum(self.train_mask))
            self.num_val = int(torch.sum(self.val_mask))
            self.num_test = int(torch.sum(self.test_mask))
            self.num_nodes = subgraph.number_of_nodes()
            self.num_edges = subgraph.number_of_edges() // 2
            self.num_features = dataset.data.num_features
            self.num_classes = dataset.data.num_classes

        if verbose:
            print("Dataset \'{}\' loaded.".format(name))
            print("    Number of nodes: {}".format(self.num_nodes))
            print("    Number of edges: {}".format(self.num_edges))
            print("    Number of features: {}".format(self.num_features))
            print("    Number of classes: {}".format(self.num_classes))
            print("    Number of train samples: {}".format(self.num_train))
            print("    Number of val samples: {}".format(self.num_val))
            print("    Number of test samples: {}".format(self.num_test))
            print("    Feature range: [{:.4f}, {:.4f}]".format(self.features.min(), self.features.max()))

    @staticmethod
    def build_adj(attr, edge_index, adj_type='csr'):
        if type(attr) == torch.Tensor:
            attr = attr.numpy()
        if type(edge_index) == torch.Tensor:
            edge_index = edge_index.numpy()
        if adj_type == 'csr':
            adj = sp.csr_matrix((attr, edge_index))
        elif adj_type == 'coo':
            adj = sp.coo_matrix((attr, edge_index))

        return adj

    COGDL_DATASETS = {
        "kdd_icdm": "cogdl.datasets.gcc_data",
        "sigir_cikm": "cogdl.datasets.gcc_data",
        "sigmod_icde": "cogdl.datasets.gcc_data",
        "usa-airport": "cogdl.datasets.gcc_data",
        "test_small": "cogdl.datasets.test_data",
        "ogbn-arxiv": "cogdl.datasets.ogb",
        "ogbn-products": "cogdl.datasets.ogb",
        "ogbn-proteins": "cogdl.datasets.ogb",
        "ogbn-mag": "cogdl.datasets.pyg_ogb",
        "ogbn-papers100M": "cogdl.datasets.ogb",
        "ogbg-molbace": "cogdl.datasets.ogb",
        "ogbg-molhiv": "cogdl.datasets.ogb",
        "ogbg-molpcba": "cogdl.datasets.ogb",
        "ogbg-ppa": "cogdl.datasets.ogb",
        "ogbg-code": "cogdl.datasets.ogb",
        "amazon": "cogdl.datasets.gatne",
        "twitter": "cogdl.datasets.gatne",
        "youtube": "cogdl.datasets.gatne",
        "gtn-acm": "cogdl.datasets.gtn_data",
        "gtn-dblp": "cogdl.datasets.gtn_data",
        "gtn-imdb": "cogdl.datasets.gtn_data",
        "fb13": "cogdl.datasets.kg_data",
        "fb15k": "cogdl.datasets.kg_data",
        "fb15k237": "cogdl.datasets.kg_data",
        "wn18": "cogdl.datasets.kg_data",
        "wn18rr": "cogdl.datasets.kg_data",
        "fb13s": "cogdl.datasets.kg_data",
        "cora": "cogdl.datasets.planetoid_data",
        "citeseer": "cogdl.datasets.planetoid_data",
        "pubmed": "cogdl.datasets.planetoid_data",
        "blogcatalog": "cogdl.datasets.matlab_matrix",
        "flickr-ne": "cogdl.datasets.matlab_matrix",
        "dblp-ne": "cogdl.datasets.matlab_matrix",
        "youtube-ne": "cogdl.datasets.matlab_matrix",
        "wikipedia": "cogdl.datasets.matlab_matrix",
        "ppi-ne": "cogdl.datasets.matlab_matrix",
        "han-acm": "cogdl.datasets.han_data",
        "han-dblp": "cogdl.datasets.han_data",
        "han-imdb": "cogdl.datasets.han_data",
        "mutag": "cogdl.datasets.tu_data",
        "imdb-b": "cogdl.datasets.tu_data",
        "imdb-m": "cogdl.datasets.tu_data",
        "collab": "cogdl.datasets.tu_data",
        "proteins": "cogdl.datasets.tu_data",
        "reddit-b": "cogdl.datasets.tu_data",
        "reddit-multi-5k": "cogdl.datasets.tu_data",
        "reddit-multi-12k": "cogdl.datasets.tu_data",
        "ptc-mr": "cogdl.datasets.tu_data",
        "nci1": "cogdl.datasets.tu_data",
        "nci109": "cogdl.datasets.tu_data",
        "enzymes": "cogdl.datasets.tu_data",
        "yelp": "cogdl.datasets.saint_data",
        "amazon-s": "cogdl.datasets.saint_data",
        "flickr": "cogdl.datasets.saint_data",
        "reddit": "cogdl.datasets.saint_data",
        "ppi": "cogdl.datasets.saint_data",
        "ppi-large": "cogdl.datasets.saint_data",
        "test_bio": "cogdl.datasets.strategies_data",
        "test_chem": "cogdl.datasets.strategies_data",
        "bio": "cogdl.datasets.strategies_data",
        "chem": "cogdl.datasets.strategies_data",
        "bace": "cogdl.datasets.strategies_data",
        "bbbp": "cogdl.datasets.strategies_data",
    }


class CustomDataset(object):
    def __init__(self, adj, features, labels, train_mask=None, val_mask=None, test_mask=None,
                 name=None, data_dir=None, mode='normal', feat_norm=None, save=False, verbose=True):
        self.adj = adj
        self.num_nodes = features.shape[0]
        self.num_edges = adj.getnnz() // 2
        self.num_features = features.shape[1]
        self.mode = mode

        if type(features) != torch.Tensor:
            features = torch.FloatTensor(features)
        elif features.type() != 'torch.FloatTensor':
            features = features.float()
        if feat_norm is not None:
            features = feat_normalize(features, norm=feat_norm)
        self.features = features

        if type(labels) != torch.Tensor:
            labels = torch.LongTensor(labels)
        elif labels.type() != 'torch.LongTensor':
            labels = labels.long()
        self.labels = labels

        if (train_mask is None) or (val_mask is None) or (test_mask is None):
            index = splitting(adj)
            self.index = index
        if train_mask is None:
            index_train = index.get("index_train")
            train_mask = torch.zeros(self.num_nodes, dtype=bool)
            train_mask[index_train] = True
        else:
            if type(train_mask) != torch.Tensor:
                train_mask = torch.BoolTensor(train_mask)
            elif train_mask.type() != 'torch.BoolTensor':
                train_mask = train_mask.bool()
        self.train_mask = train_mask

        if val_mask is None:
            index_val = index.get("index_val")
            val_mask = torch.zeros(self.num_nodes, dtype=bool)
            val_mask[index_val] = True
        else:
            if type(val_mask) != torch.Tensor:
                val_mask = torch.BoolTensor(val_mask)
            elif val_mask.type() != 'torch.BoolTensor':
                val_mask = val_mask.bool()
        self.val_mask = val_mask

        if test_mask is None:
            if mode == "easy":
                index_test = index.get("index_test_easy")
            elif mode == "medium":
                index_test = index.get("index_test_medium")
            elif mode == "hard":
                index_test = index.get("index_test_hard")
            elif mode == "normal":
                index_test = index.get("index_test")
            else:
                index_test = index.get("index_test")
            test_mask = torch.zeros(self.num_nodes, dtype=bool)
            test_mask[index_test] = True
        else:
            if type(test_mask) != torch.Tensor:
                test_mask = torch.BoolTensor(test_mask)
            elif test_mask.type() != 'torch.BoolTensor':
                test_mask = test_mask.bool()
        self.test_mask = test_mask

        self.num_train = int(torch.sum(self.train_mask))
        self.num_val = int(torch.sum(self.val_mask))
        self.num_test = int(torch.sum(self.test_mask))
        if len(labels.shape) == 1:
            self.num_classes = int(labels.max() + 1)
        else:
            self.num_classes = labels.shape[-1]

        if save:
            if data_dir is None:
                data_dir = "./data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            sp.save_npz(os.path.join(data_dir, "adj.npz"), adj.tocsr())
            np.savez_compressed(os.path.join(data_dir, "index.npz"), **index)
            np.savez_compressed(os.path.join(data_dir, "features.npz"), data=features)
            np.savez_compressed(os.path.join(data_dir, "labels.npz"), data=labels)
            print("    Saved in {}.".format(data_dir))

        if verbose:
            print("Custom Dataset \'{}\' loaded.".format(name))
            print("    Number of nodes: {}".format(self.num_nodes))
            print("    Number of edges: {}".format(self.num_edges))
            print("    Number of features: {}".format(self.num_features))
            print("    Number of classes: {}".format(self.num_classes))
            print("    Number of train samples: {}".format(self.num_train))
            print("    Number of val samples: {}".format(self.num_val))
            print("    Number of test samples: {}".format(self.num_test))
            print("    Dataset mode: {}".format(self.mode))
            print("    Feature range [{:.4f}, {:.4f}]".format(self.features.min(), self.features.max()))


def feat_normalize(feat, norm=None, lim_min=-1, lim_max=1):
    if norm == "linearize":
        k = (lim_max - lim_min) / (feat.max() - feat.min())
        feat = lim_min + k * (feat - feat.min())
    elif norm == "arctan":
        feat = (feat - feat.mean()) / feat.std()
        feat = 2 * np.arctan(feat) / np.pi
    elif norm == "tanh":
        feat = (feat - feat.mean()) / feat.std()
        feat = np.tanh(feat)
    elif norm == "standardize":
        feat = (feat - feat.mean()) / feat.std()
    else:
        feat = feat

    return feat


def splitting(adj,
              range_min=(0.0, 0.05),
              range_max=(0.95, 1.0),
              range_easy=(0.05, 0.35),
              range_medium=(0.35, 0.65),
              range_hard=(0.65, 0.95),
              ratio_train=0.6,
              ratio_val=0.1,
              ratio_test=0.1,
              seed=42):
    def a_not_in_b(a, b):
        c = []
        for i in a:
            if i not in b:
                c.append(i)

        return np.array(c)

    num_nodes = adj.shape[0]
    degs = adj.getnnz(axis=1)
    print("GRB data splitting...")
    print("    Average degree of all nodes: {:.4f}".format(np.mean(degs)))

    degs_index = np.argsort(degs)
    ind_min = int(len(degs_index) * range_min[1])
    ind_max = int(len(degs_index) * range_max[0])
    print("    Average degree of 5% nodes with small degree: {:.4f}".format(
        np.mean(degs[degs_index[:ind_min]])))
    print("    Average degree of 5% nodes with large degree: {:.4f}".format(
        np.mean(degs[degs_index[ind_max:]])))

    # Sampling 'easy' test nodes
    ind_easy_min = int(len(degs_index) * range_easy[0])
    ind_easy_max = int(len(degs_index) * range_easy[1])
    print("    Average degree of 30% nodes (easy): {:.4f}".format(
        np.mean(degs[degs_index[ind_easy_min:ind_easy_max]])))

    np.random.seed(seed)
    ind_easy_sample = np.random.choice(degs_index[ind_easy_min:ind_easy_max],
                                       int(num_nodes * ratio_test), replace=False)
    print("    Randomly sampled {} nodes".format(ind_easy_sample.shape[0]))

    # Sampling 'medium' test nodes
    ind_medium_min = int(len(degs_index) * range_medium[0])
    ind_medium_max = int(len(degs_index) * range_medium[1])
    print("    Average degree of 30% nodes (medium): {:.4f}".format(
        np.mean(degs[degs_index[ind_medium_min:ind_medium_max]])))

    np.random.seed(seed)
    ind_medium_sample = np.random.choice(degs_index[ind_medium_min:ind_medium_max],
                                         int(num_nodes * ratio_test), replace=False)
    print("    Randomly sampled {} nodes".format(ind_medium_sample.shape[0]))

    # Sampling 'hard' test nodes
    ind_hard_min = int(len(degs_index) * range_hard[0])
    ind_hard_max = int(len(degs_index) * range_hard[1])
    print("    Average degree of 30% nodes (hard): {:.4f}".format(
        np.mean(degs[degs_index[ind_hard_min:ind_hard_max]])))

    np.random.seed(seed)
    ind_hard_sample = np.random.choice(degs_index[ind_hard_min:ind_hard_max],
                                       int(num_nodes * ratio_test), replace=False)
    print("    Randomly sampled {} nodes".format(ind_hard_sample.shape[0]))

    ind_test = np.concatenate([ind_easy_sample,
                               ind_medium_sample,
                               ind_hard_sample])

    # Sampling nodes for training and validation
    ind_rest = a_not_in_b(degs_index, ind_test)
    np.random.seed(seed)
    ind_train = np.random.choice(ind_rest, int(num_nodes * ratio_train), replace=False)
    ind_val = a_not_in_b(ind_rest, ind_train)
    print("    Number of training/validation nodes: {}/{}".format(len(ind_train), len(ind_val)))

    if len(ind_train) + len(ind_val) + len(ind_test) == num_nodes:
        print("    No duplicate.")
    else:
        print("    Find duplicates.")

    index = {"index_train": np.sort(ind_train),
             "index_val": np.sort(ind_val),
             "index_test": np.sort(ind_test),
             "index_test_easy": np.sort(ind_easy_sample),
             "index_test_medium": np.sort(ind_medium_sample),
             "index_test_hard": np.sort(ind_hard_sample)}

    return index
