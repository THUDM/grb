import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import grb.utils as utils
from grb.attack.base import InjectionAttack
from grb.utils import evaluator


class RND(InjectionAttack):
    def __init__(self, dataset, adj_norm_func=None, device='cpu'):
        self.dataset = dataset
        self.n_total = dataset.num_nodes
        self.n_test = dataset.num_test
        self.n_feat = dataset.num_features
        self.n_inject_max = 0
        self.n_edge_max = 0
        self.adj_norm_func = adj_norm_func
        self.device = device
        self.config = {}

    def set_config(self, **kwargs):
        self.config['feat_lim_min'] = kwargs['feat_lim_min']
        self.config['feat_lim_max'] = kwargs['feat_lim_max']
        self.n_inject_max = kwargs['n_inject_max']
        self.n_edge_max = kwargs['n_edge_max']

    def attack(self, model):
        model.to(self.device)
        features = self.dataset.features
        features = torch.FloatTensor(features).to(self.device)
        adj = self.dataset.adj
        adj_tensor = utils.adj_preprocess(adj, adj_norm_func=self.adj_norm_func, device=self.device)
        pred_orig = model(features, adj_tensor)
        origin_labels = torch.argmax(pred_orig, dim=1)
        adj_attack = self.injection(adj=adj,
                                    n_inject=self.n_inject_max,
                                    n_node=self.n_total)

        features_attack = np.zeros((self.n_inject_max, self.n_feat))
        features_attack = self.update_features(model=model,
                                               adj_attack=adj_attack,
                                               features=features,
                                               features_attack=features_attack,
                                               origin_labels=origin_labels)

        return adj_attack, features_attack

    def injection(self, adj, n_inject, n_node):
        test_index = torch.where(self.dataset.test_mask)[0]
        n_test = test_index.shape[0]
        new_edges_x = []
        new_edges_y = []
        new_data = []
        for i in range(n_inject):
            islinked = np.zeros(self.n_test)
            for j in range(n_inject):
                x = i + n_node

                yy = random.randint(0, n_test - 1)
                while islinked[yy] > 0:
                    yy = random.randint(0, n_test - 1)

                y = test_index[yy]
                new_edges_x.extend([x, y])
                new_edges_y.extend([y, x])
                new_data.extend([1, 1])

        add1 = sp.csr_matrix((n_inject, n_node))
        add2 = sp.csr_matrix((n_node + n_inject, n_inject))
        adj_attack = sp.vstack([adj, add1])
        adj_attack = sp.hstack([adj_attack, add2])
        adj_attack.row = np.hstack([adj_attack.row, new_edges_x])
        adj_attack.col = np.hstack([adj_attack.col, new_edges_y])
        adj_attack.data = np.hstack([adj_attack.data, new_data])

        return adj_attack

    def update_features(self, model, adj_attack, features, features_attack, origin_labels):
        feat_lim_min, feat_lim_max = self.config['feat_lim_min'], self.config['feat_lim_max']

        adj_attacked_tensor = utils.adj_preprocess(adj_attack, adj_norm_func=self.adj_norm_func, device=self.device)
        features_attack = np.random.normal(loc=0, scale=self.config['feat_lim_max'],
                                           size=(self.n_inject_max, self.n_feat))
        features_attack = np.clip(features_attack, feat_lim_min, feat_lim_max)
        features_attack = torch.FloatTensor(features_attack).to(self.device)
        model.eval()

        features_concat = torch.cat((features, features_attack), dim=0)
        pred = model(features_concat, adj_attacked_tensor)
        pred_loss = -F.nll_loss(pred[:self.n_total][self.dataset.test_mask],
                                origin_labels[self.dataset.test_mask]).to(self.device)

        test_acc = evaluator.eval_acc(pred[:self.n_total][self.dataset.test_mask],
                                      origin_labels[self.dataset.test_mask])

        print("Loss: {:.5f}, Test acc: {:.5f}".format(pred_loss, test_acc))

        return features_attack
