import dgl
import networkx
import torch

from grb.utils import utils


class DataLoader(object):
    def __init__(self, name):
        if name == 'cora':
            dataset = dgl.data.CoraGraphDataset()
            graph = dataset.graph
            self.adj = networkx.convert_matrix.to_scipy_sparse_matrix(graph, format='coo')
            self.features = dataset.features
            self.labels = dataset.labels
            self.train_mask = dataset.train_mask
            self.val_mask = dataset.val_mask
            self.test_mask = dataset.test_mask
            self.num_classes = dataset.num_classes
            self.adj_tensor = None
            self.preprocess()

        else:
            raise NotImplementedError

    def preprocess(self):
        self.adj_tensor = utils.adj_to_tensor(self.adj)
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.LongTensor(self.labels)
