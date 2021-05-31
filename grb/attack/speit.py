import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import grb.utils as utils
from grb.attack.base import InjectionAttack
from grb.evaluator import metric


class SPEIT(InjectionAttack):
    def __init__(self,
                 lr,
                 n_epoch,
                 n_inject_max,
                 n_edge_max,
                 feat_lim_min,
                 feat_lim_max,
                 loss=F.nll_loss,
                 eval_metric=metric.eval_acc,
                 inject_mode='random',
                 device='cpu',
                 early_stop=None,
                 verbose=True):
        self.device = device
        self.lr = lr
        self.n_epoch = n_epoch
        self.n_inject_max = n_inject_max
        self.n_edge_max = n_edge_max
        self.feat_lim_min = feat_lim_min
        self.feat_lim_max = feat_lim_max
        self.loss = loss
        self.eval_metric = eval_metric
        self.inject_mode = inject_mode
        self.verbose = verbose

        # Early stop
        if early_stop:
            self.early_stop = EarlyStop(patience=1000, epsilon=1e-4)
        else:
            self.early_stop = early_stop

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
                                    target_mask=target_mask,
                                    mode=self.inject_mode)

        features_attack = np.zeros([self.n_inject_max, n_feat])
        features_attack = self.update_features(model=model,
                                               adj_attack=adj_attack,
                                               features=features,
                                               features_attack=features_attack,
                                               origin_labels=origin_labels,
                                               target_mask=target_mask,
                                               adj_norm_func=adj_norm_func)

        return adj_attack, features_attack

    def injection(self, adj, n_inject, n_node, target_mask, target_node=None, mode='random-inter'):
        def get_inject_list(adj, n_inter, inject_id, inject_tmp_list):
            i = 1
            res = []
            while len(res) < n_inter and i < len(inject_tmp_list):
                if adj[inject_id, inject_tmp_list[i]] == 0:
                    res.append(inject_tmp_list[i])
                i += 1

            return res

        def update_active_edges(active_edges, n_inject_edges, threshold):
            for i in active_edges:
                if n_inject_edges[i] >= threshold:
                    active_edges.pop(active_edges.index(i))

            return active_edges

        def inject(target_node_list, n_inject, n_test, mode='random-inter'):
            n_target = len(target_node_list)
            adj = np.zeros((n_inject, n_test + n_inject))

            if mode == 'random-inter':
                # target_node_list: a list of target nodes to be attacked
                n_inject_edges = np.zeros(n_inject)
                active_edges = [i for i in range(n_inject)]

                # create edges between injected node and target node
                for i in range(n_target):
                    if not active_edges:
                        break
                    inject_id = np.random.choice(active_edges, 1)
                    n_inject_edges[inject_id] += 1
                    active_edges = update_active_edges(active_edges, n_inject_edges, threshold=self.n_edge_max)
                    adj[inject_id, target_node_list[i]] = 1

                # create edges between injected nodes
                for i in range(len(active_edges)):
                    if not active_edges:
                        break
                    inject_tmp_list = sorted(active_edges, key=lambda x: n_inject_edges[x])
                    inject_id = inject_tmp_list[0]
                    n_inter = self.n_edge_max - n_inject_edges[inject_id]
                    inject_list = get_inject_list(adj, n_inter, inject_id, inject_tmp_list)

                    n_inject_edges[inject_list] += 1
                    n_inject_edges[inject_id] += len(inject_list)
                    active_edges = update_active_edges(active_edges, n_inject_edges, threshold=self.n_edge_max)
                    if inject_list:
                        adj[inject_id, n_test + np.array(inject_list)] = 1
                        adj[inject_list, n_test + inject_id] = 1

            elif mode == 'multi-layer':
                n_inject_edges = np.zeros(n_inject)
                n_inject_layer_1, n_inject_layer_2 = int(n_inject * 0.9), int(n_inject * 0.1)
                n_edge_max_layer_1 = int(self.n_edge_max * 0.9)
                active_edges = [i for i in range(n_inject_layer_1)]

                # create edges between noise node and test node
                for i in range(n_target):
                    if not active_edges:
                        break
                    inject_id = np.random.choice(active_edges, 1)
                    n_inject_edges[inject_id] += 1
                    active_edges = update_active_edges(active_edges, n_inject_edges, threshold=n_edge_max_layer_1)
                    adj[inject_id, target_node_list[i]] = 1

                # create edges between injected nodes
                for i in range(len(active_edges)):
                    if not active_edges:
                        break
                    inject_tmp_list = sorted(active_edges, key=lambda x: n_inject_edges[x])
                    inject_id = inject_tmp_list[0]
                    n_inter = n_edge_max_layer_1 - n_inject_edges[inject_id]
                    inject_list = get_inject_list(adj, n_inter, inject_id, inject_tmp_list)

                    n_inject_edges[inject_list] += 1
                    n_inject_edges[inject_id] += len(inject_list)

                    active_edges = update_active_edges(active_edges, n_inject_edges, threshold=n_edge_max_layer_1)

                    if inject_list:
                        adj[inject_id, n_test + np.array(inject_list)] = 1
                        adj[inject_list, n_test + inject_id] = 1

                noise_active_layer2 = [i for i in range(n_inject_layer_2)]
                noise_edge_layer2 = np.zeros(n_inject_layer_2)
                for i in range(n_inject_layer_1):
                    if not noise_active_layer2:
                        break
                    inject_list = np.random.choice(noise_active_layer2, 10)
                    noise_edge_layer2[inject_list] += 1
                    noise_active_layer2 = update_active_edges(noise_active_layer2, noise_edge_layer2,
                                                              threshold=self.n_edge_max)
                    adj[inject_list + n_inject_layer_1, i + n_test] = 1
                    adj[i, inject_list + n_inject_layer_1 + n_test] = 1
            else:
                print("Mode ERROR: 'mode' should be one of ['random', 'random-inter', 'multi-layer']")

            return adj

        if mode == 'random':
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
        else:
            # construct injected adjacency matrix
            test_index = torch.where(target_mask)[0]
            n_test = test_index.shape[0]
            adj_inject = inject(target_node, n_inject, n_test, mode)
            adj_inject = sp.hstack([sp.csr_matrix((n_inject, n_node - n_test)), adj_inject])
            adj_inject = sp.csr_matrix(adj_inject)
            adj_attack = sp.vstack([adj, adj_inject[:, :n_node]])
            adj_attack = sp.hstack([adj_attack, adj_inject.T])
            adj_attack = sp.coo_matrix(adj_attack)

            return adj_attack

    def update_features(self, model, adj_attack, features, features_attack, origin_labels, target_mask, adj_norm_func):
        lr = self.lr
        n_epoch = self.n_epoch
        feat_lim_min, feat_lim_max = self.feat_lim_min, self.feat_lim_max

        n_total = features.shape[0]
        adj_attacked_tensor = utils.adj_preprocess(adj=adj_attack,
                                                   adj_norm_func=adj_norm_func,
                                                   model_type=model.model_type,
                                                   device=self.device)
        features_attack = utils.feat_preprocess(features=features_attack, device=self.device)
        features_attack.requires_grad_(True)
        optimizer = torch.optim.Adam([features_attack], lr=lr)
        model.eval()

        for i in range(n_epoch):
            features_concat = torch.cat((features, features_attack), dim=0)
            pred = model(features_concat, adj_attacked_tensor)
            pred_loss = -F.nll_loss(pred[:n_total][target_mask],
                                    origin_labels[target_mask]).to(self.device)
            optimizer.zero_grad()
            pred_loss.backward(retain_graph=True)
            optimizer.step()

            with torch.no_grad():
                features_attack.clamp_(feat_lim_min, feat_lim_max)

            test_score = metric.eval_acc(pred[:n_total][target_mask],
                                         origin_labels[target_mask])

            if self.early_stop:
                self.early_stop(test_score)
                if self.early_stop.stop:
                    print("Attacking: Early stopped.")
                    self.early_stop = EarlyStop()
                    return features_attack

            if self.verbose:
                print(
                    "Attacking: Epoch {}, Loss: {:.5f}, Surrogate test score: {:.5f}".format(i, pred_loss, test_score),
                    end='\r' if i != n_epoch - 1 else '\n')

        return features_attack


class EarlyStop(object):
    def __init__(self, patience=1000, epsilon=1e-4):
        self.patience = patience
        self.epsilon = epsilon
        self.min_score = None
        self.stop = False
        self.count = 0

    def __call__(self, score):
        if self.min_score is None:
            self.min_score = score
        elif self.min_score - score > 0:
            self.count = 0
            self.min_score = score
        elif self.min_score - score < self.epsilon:
            self.count += 1
            if self.count > self.patience:
                self.stop = True
