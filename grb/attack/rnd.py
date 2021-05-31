import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import grb.utils as utils
from grb.attack.base import InjectionAttack
from grb.evaluator import metric


class RND(InjectionAttack):
    def __init__(self,
                 n_inject_max,
                 n_edge_max,
                 feat_lim_min,
                 feat_lim_max,
                 loss=F.nll_loss,
                 eval_metric=metric.eval_acc,
                 device='cpu',
                 verbose=True):
        self.device = device
        self.n_inject_max = n_inject_max
        self.n_edge_max = n_edge_max
        self.feat_lim_min = feat_lim_min
        self.feat_lim_max = feat_lim_max
        self.loss = loss
        self.eval_metric = eval_metric
        self.verbose = verbose

    def attack(self, model, adj, features, target_mask, adj_norm_func):
        model.to(self.device)
        n_total, n_feat = features.shape
        features = utils.feat_preprocess(features=features, device=self.device)
        adj_tensor = utils.adj_preprocess(adj=adj,
                                          adj_norm_func=adj_norm_func,
                                          model_type=model.model_type,
                                          device=self.device)
        pred_orig = model(features, adj_tensor)
        origin_labels = torch.argmax(pred_orig, dim=1)
        adj_attack = self.injection(adj=adj,
                                    n_inject=self.n_inject_max,
                                    n_node=n_total,
                                    target_mask=target_mask)

        features_attack = np.zeros((self.n_inject_max, n_feat))
        features_attack = self.update_features(model=model,
                                               adj_attack=adj_attack,
                                               features=features,
                                               features_attack=features_attack,
                                               origin_labels=origin_labels,
                                               target_mask=target_mask,
                                               adj_norm_func=adj_norm_func)

        return adj_attack, features_attack

    def injection(self, adj, n_inject, n_node, target_mask):
        test_index = torch.where(target_mask)[0]
        n_test = test_index.shape[0]
        new_edges_x = []
        new_edges_y = []
        new_data = []
        for i in range(n_inject):
            islinked = np.zeros(n_test)
            for j in range(self.n_edge_max):
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

    def update_features(self, model, adj_attack, features, features_attack, origin_labels, target_mask, adj_norm_func):
        feat_lim_min, feat_lim_max = self.feat_lim_min, self.feat_lim_max
        n_total = features.shape[0]

        adj_attacked_tensor = utils.adj_preprocess(adj=adj_attack,
                                                   adj_norm_func=adj_norm_func,
                                                   model_type=model.model_type,
                                                   device=self.device)
        features_attack = np.random.normal(loc=0, scale=feat_lim_max,
                                           size=(self.n_inject_max, features.shape[1]))
        features_attack = np.clip(features_attack, feat_lim_min, feat_lim_max)
        features_attack = utils.feat_preprocess(features=features_attack, device=self.device)
        model.eval()

        features_concat = torch.cat((features, features_attack), dim=0)
        pred = model(features_concat, adj_attacked_tensor)
        pred_loss = -self.loss(pred[:n_total][target_mask],
                               origin_labels[target_mask]).to(self.device)

        test_acc = self.eval_metric(pred[:n_total][target_mask],
                                    origin_labels[target_mask])

        print("Loss: {:.5f}, Surrogate test acc: {:.5f}".format(pred_loss, test_acc))

        return features_attack
