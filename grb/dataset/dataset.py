import os

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from cogdl.datasets import build_dataset_from_name, build_dataset_from_path

from ..dataset import URLs, SUPPORTED_DATASETS
from ..utils import download


class Dataset(object):
    r"""

    Description
    -----------
    Class that loads GRB datasets for evaluating adversarial robustness.

    Parameters
    ----------
    name: str
        Name of dataset, supported datasets: ``["grb-cora", "grb-citeseer", "grb-aminer", "grb-reddit", "grb-flickr"]``.
    data_dir: str, optional
        Directory for dataset. If not provided, default is ``"./data/"``.
    mode: str, optional
        Difficulty determined according to the average degree of test nodes.
        Choose from ``["easy", "medium", "hard", "full"]``. Default: ``"full"`` is to use the entire test set.
    feat_norm: str, optional
        Feature normalization that transform all features to range [-1, 1].
        Choose from ``["arctan", "sigmoid", "tanh"]``. Default: ``None``.
    verbose: bool, optional
        Whether to display logs. Default: ``True``.

    Attributes
    ----------

    adj : scipy.sparse.csr.csr_matrix
        Adjacency matrix in form of ``N * N`` sparse matrix.
    features : torch.FloatTensor
        Features in form of ``N * D`` torch float tensor.
    labels : torch.LongTensor
        Labels in form of ``N * L``. L=1 for multi-class classification, otherwise for multi-label classification.
    num_nodes: int
        Number of nodes ``N``.
    num_edges: int
        Number of edges.
    num_features: int
        Dimension of features ``D``.
    num_classes : int
        Number of classes ``L``.
    num_train : int
        Number of train nodes.
    num_val: int
        Number of validation nodes.
    num_test: int
        Number of test nodes.
    mode: str
        Mode of dataset. One of ``["easy", "medium", "hard", "full"]``.
    index_train: np.ndarray
        Index of train nodes.
    index_val: np.ndarray
        Index of validation nodes.
    index_test: np.ndarray
        Index of test nodes.
    train_mask: torch.Tensor
        Mask of train nodes in form of ``N * 1`` torch bool tensor.
    val_mask : torch.Tensor
        Mask of validation nodes in form of ``N * 1`` torch bool tensor.
    test_mask : torch.Tensor
        Mask of test nodes in form of ``N * 1`` torch bool tensor.

    Example
    -------
    >>> import grb
    >>> from grb.dataset import Dataset
    >>> dataset = Dataset(name='grb-cora', mode='easy', feat_norm="arctan")

    """

    def __init__(self, name, data_dir=None, mode="easy", feat_norm=None, verbose=True):
        # Create data dir
        if name not in SUPPORTED_DATASETS:
            print("{} dataset not supported.".format(name))
            exit(1)
        if data_dir is None:
            data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", name)
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


class CogDLDataset(object):
    def __init__(self, name, data_dir=None, mode='origin', verbose=True):
        r"""

        Description
        -----------
        Class that loads `CogDL datasets <https://github.com/THUDM/cogdl/tree/master/cogdl/datasets>`__
        for GRB evaluation.

        Parameters
        ----------
        name: str
            Name of dataset, see supported datasets in self.COGDL_DATASETS.
        data_dir: str, optional
            Directory for dataset. If not provided, default is ``"./data/"``.
        mode: str, optional
            Choose from ``["original", "lcc"]``. ``lcc`` is to extract the largest connected components.
            Default: ``original``.
        verbose: bool, optional
            Whether to display logs. Default: ``True``.
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


class CustomDataset(object):
    r"""

    Description
    -----------
    Class that helps to build customized dataset for GRB evaluation.

    Parameters
    ----------
    adj : scipy.sparse.csr.csr_matrix
        Adjacency matrix in form of ``N * N`` sparse matrix.
    features : torch.FloatTensor
        Features in form of ``N * D`` torch float tensor.
    labels : torch.LongTensor
        Labels in form of ``N * L``. L=1 for multi-class classification, otherwise for multi-label classification.
    train_mask: torch.Tensor, optional
        Mask of train nodes in form of ``N * 1`` torch bool tensor. Default: ``None``.
        If is ``None``, generated by default splitting scheme.
    val_mask : torch.Tensor, optional
        Mask of validation nodes in form of ``N * 1`` torch bool tensor. Default: ``None``.
        If is ``None``, generated by default splitting scheme.
    test_mask : torch.Tensor, optional
        Mask of test nodes in form of ``N * 1`` torch bool tensor. Default: ``None``.
        If is ``None``, generated by default splitting scheme.
    name : str, optional
        Name of dataset.
    data_dir : str, optional
        Directory of dataset.
    mode : str, optional
        Mode of dataset. One of ``["easy", "medium", "hard", "full"]``. Default: ``full``.
    feat_norm : str, optional
        Feature normalization that transform all features to range [-1, 1].
        Choose from ``["arctan", "sigmoid", "tanh"]``. Default: ``None``.
    save : bool, optional
        Whether to save data as files.
    verbose : bool, optional
        Whether to display logs. Default: ``True``.

    Parameters
    ----------
    name: str
        Name of dataset, supported datasets: ``["grb-cora", "grb-citeseer", "grb-aminer", "grb-reddit", "grb-flickr"]``.
    data_dir: str, optional
        Directory for dataset. If not provided, default is ``"./data/"``.
    mode: str, optional
        Difficulty determined according to the average degree of test nodes.
        Choose from ``["easy", "medium", "hard", "full"]``. Default: ``"full"`` is to use the entire test set.
    feat_norm: str, optional
        Feature normalization that transform all features to range [-1, 1].
        Choose from ``["arctan", "sigmoid", "tanh"]``. Default: ``None``.
    verbose: bool, optional
        Whether to display logs. Default: ``True``.

    Attributes
    ----------

    adj : scipy.sparse.csr.csr_matrix
        Adjacency matrix in form of ``N * N`` sparse matrix.
    features : torch.FloatTensor
        Features in form of ``N * D`` torch float tensor.
    labels : torch.LongTensor
        Labels in form of ``N * L``. L=1 for multi-class classification, otherwise for multi-label classification.
    num_nodes: int
        Number of nodes ``N``.
    num_edges: int
        Number of edges.
    num_features: int
        Dimension of features ``D``.
    num_classes : int
        Number of classes ``L``.
    num_train : int
        Number of train nodes.
    num_val: int
        Number of validation nodes.
    num_test: int
        Number of test nodes.
    mode: str
        Mode of dataset. One of ``["easy", "medium", "hard", "full"]``.
    index_train: np.ndarray
        Index of train nodes.
    index_val: np.ndarray
        Index of validation nodes.
    index_test: np.ndarray
        Index of test nodes.
    train_mask: torch.Tensor
        Mask of train nodes in form of ``N * 1`` torch bool tensor.
    val_mask : torch.Tensor
        Mask of validation nodes in form of ``N * 1`` torch bool tensor.
    test_mask : torch.Tensor
        Mask of test nodes in form of ``N * 1`` torch bool tensor.

    """
    def __init__(self, adj, features, labels, train_mask=None, val_mask=None, test_mask=None,
                 name=None, data_dir=None, mode='full', feat_norm=None, save=False, verbose=True):
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
            elif mode == "full":
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


def feat_normalize(features, norm=None, lim_min=-1.0, lim_max=1.0):
    r"""
    Description
    -----------
    Feature normalization function.

    Parameters
    ----------
    features : torch.FloatTensor
        Features in form of ``N * D`` torch float tensor.
    norm : str, optional
        Type of normalization. Choose from ``["linearize", "arctan", "tanh", "standarize"]``.
        Default: ``None``.
    lim_min : float
        Minimum limit of feature value. Default: ``-1.0``.
    lim_max : float
        Minimum limit of feature value. Default: ``1.0``.

    Returns
    -------
    features : torch.FloatTensor
        Normalized features in form of ``N * D`` torch float tensor.

    """
    if norm == "linearize":
        k = (lim_max - lim_min) / (features.max() - features.min())
        features = lim_min + k * (features - features.min())
    elif norm == "arctan":
        features = (features - features.mean()) / features.std()
        features = 2 * np.arctan(features) / np.pi
    elif norm == "tanh":
        features = (features - features.mean()) / features.std()
        features = np.tanh(features)
    elif norm == "standardize":
        features = (features - features.mean()) / features.std()
    else:
        features = features

    return features


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
    r"""

    Description
    -----------
    GRB splitting scheme designed for adversarial robustness evaluation.

    Parameters
    ----------
    adj : scipy.sparse.csr.csr_matrix
        Adjacency matrix in form of ``N * N`` sparse matrix.
    range_min : tuple of float, optional
        Range of nodes with minimum degrees to be ignored. Value in percentage.
        Default: ``(0.0, 0.05)``.
    range_max : tuple of float, optional
        Range of nodes with maximum degrees to be ignored. Value in percentage.
        Default: ``(0.95, 1.0)``.
    range_easy : tuple of float, optional
        Range of nodes for ``easy`` difficulty. Value in percentage.
        Default: ``(0.05, 0.35)``.
    range_medium : tuple of float, optional
        Range of nodes for ``medium`` difficulty. Value in percentage.
        Default: ``(0.35, 0.65)``.
    range_hard : tuple of float, optional
        Range of nodes for ``hard`` difficulty. Value in percentage.
        Default: ``(0.65, 0.95)``.
    ratio_train : float, optional
        Ratio of train nodes. Default: ``0.6``.
    ratio_val : float, optional
        Ratio of validation nodes. Default: ``0.1``.
    ratio_test : float, optional
        Ratio of test nodes. Default: ``0.1``.
    seed : int, optional
        Random seed. Default: ``42``.

    Returns
    -------
    index : dict
        Dictionary containing ``{"index_train", "index_val", "index_test",
        "index_test_easy", "index_test_medium", "index_test_hard"}``.

    """
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

    index = {"index_train"      : np.sort(ind_train),
             "index_val"        : np.sort(ind_val),
             "index_test"       : np.sort(ind_test),
             "index_test_easy"  : np.sort(ind_easy_sample),
             "index_test_medium": np.sort(ind_medium_sample),
             "index_test_hard"  : np.sort(ind_hard_sample)}

    return index
