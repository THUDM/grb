import torch
import networkx as nx

from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict

from grb.utils import adj_to_tensor


class DataLoader(object):
    def __init__(self, name, mode='normal', verbose=True):
        """mode: 'normal', 'lcc';"""
        default_dict = {"dataset": name}
        args = build_args_from_dict(default_dict)
        dataset = build_dataset(args)
        if mode == 'normal':
            self.adj = dataset.data._build_adj_().tocoo()
            self.adj_tensor = adj_to_tensor(self.adj)
            self.features = dataset.data.x
            self.labels = dataset.data.y
            self.train_mask = dataset.data.train_mask
            self.val_mask = dataset.data.val_mask
            self.test_mask = dataset.data.test_mask
            self.num_train = int(torch.sum(self.train_mask))
            self.num_val = int(torch.sum(self.val_mask))
            self.num_test = int(torch.sum(self.test_mask))
            self.num_nodes = dataset.data.num_nodes
            self.num_edges = dataset.data.num_edges
            self.num_features = dataset.data.num_features
            self.num_classes = dataset.data.num_classes
        elif mode == 'lcc':
            # Get largest connected component
            adj = dataset.data._build_adj_()
            graph = nx.from_scipy_sparse_matrix(adj)
            components = nx.connected_components(graph)
            lcc_nodes = list(next(components))
            subgraph = graph.subgraph(lcc_nodes)
            self.adj = nx.to_scipy_sparse_matrix(subgraph, format='coo')
            self.adj_tensor = adj_to_tensor(self.adj)
            self.features = dataset.data.x[lcc_nodes]
            self.labels = dataset.data.y[lcc_nodes]
            self.train_mask = dataset.data.train_mask[lcc_nodes]
            self.val_mask = dataset.data.val_mask[lcc_nodes]
            self.test_mask = dataset.data.test_mask[lcc_nodes]
            self.num_train = int(torch.sum(self.train_mask))
            self.num_val = int(torch.sum(self.val_mask))
            self.num_test = int(torch.sum(self.test_mask))
            self.num_nodes = subgraph.number_of_nodes()
            self.num_edges = subgraph.number_of_edges()
            self.num_features = dataset.data.num_features
            self.num_classes = dataset.data.num_classes

        if verbose:
            print("Dataset \'{}\' loaded.".format(name))
            print("    Number of nodes: {}.".format(self.num_nodes))
            print("    Number of edges: {}.".format(self.num_edges))
            print("    Number of features: {}.".format(self.num_features))
            print("    Number of classes: {}.".format(self.num_classes))
            print("    Number of train samples: {}.".format(self.num_train))
            print("    Number of val samples: {}.".format(self.num_val))
            print("    Number of test samples: {}.".format(self.num_test))
            print("    Feature range [{:.4f}, {:.4f}]".format(self.features.min(), self.features.max()))

    SUPPORTED_DATASETS = {
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


class CustomDataLoader(object):
    def __init__(self, adj, features, labels, train_mask, val_mask, test_mask, name=None, verbose=True):
        self.adj = adj.tocoo()
        self.adj_tensor = adj_to_tensor(self.adj)
        self.features = features
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_train = int(torch.sum(self.train_mask))
        self.num_val = int(torch.sum(self.val_mask))
        self.num_test = int(torch.sum(self.test_mask))
        self.num_nodes = features.shape[0]
        self.num_edges = adj.getnnz()
        self.num_features = features.shape[1]
        self.num_classes = labels.max() + 1

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
