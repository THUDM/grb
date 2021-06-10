import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

import grb.utils as utils
from grb.attack.base import InjectionAttack
from grb.evaluator import metric


class TDGIA(InjectionAttack):
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

    def attack(self, model, adj, features, target_mask, adj_norm_func, opt='sin'):
        model.to(self.device)
        n_total, n_feat = features.shape
        features = utils.feat_preprocess(features=features, device=self.device)
        adj_tensor = utils.adj_preprocess(adj=adj,
                                          adj_norm_func=adj_norm_func,
                                          model_type=model.model_type,
                                          device=self.device)
        pred_orig = model(features, adj_tensor)
        pred_orig_logits = F.softmax(pred_orig, dim=1)
        origin_labels = torch.argmax(pred_orig, dim=1)

        adj_attack = self.injection(adj=adj,
                                    n_inject=self.n_inject_max,
                                    n_node=n_total,
                                    target_mask=target_mask,
                                    origin_labels=origin_labels,
                                    logits=pred_orig_logits,
                                    n_add=0,
                                    mode=self.inject_mode)

        features_attack = np.zeros([self.n_inject_max, n_feat])
        features_attack = self.update_features(model=model,
                                               adj_attack=adj_attack,
                                               features=features,
                                               features_attack=features_attack,
                                               origin_labels=origin_labels,
                                               target_mask=target_mask,
                                               adj_norm_func=adj_norm_func,
                                               opt=opt)

        return adj_attack, features_attack

    def injection(self, adj, n_inject, n_node, target_mask,
                  origin_labels, logits, n_add, self_connect_ratio=0.0, mode='uniform', weight1=0.9, weight2=0.1):
        """Injection mode: uniform, random, tdgia. """
        n_inject = self.n_inject_max
        n_origin = origin_labels.shape[0]
        n_test = torch.sum(target_mask).item()
        n_classes = origin_labels.max() + 1
        n_current = n_origin + n_add
        n_connect = int(self.n_edge_max * (1 - self_connect_ratio))
        n_self_connect = int(self.n_edge_max * self_connect_ratio)

        new_edges_x = []
        new_edges_y = []
        new_data = []
        test_index = torch.where(target_mask)[0]

        if 'uniform' in mode:
            for i in range(n_inject):
                x = i + n_current
                for j in range(n_connect):
                    id = (x - n_origin) * n_connect + j
                    id = id % n_test
                    y = test_index[id]
                    new_edges_x.extend([x, y])
                    new_edges_y.extend([y, x])
                    new_data.extend([1, 1])

            add1 = sp.csr_matrix((n_inject, n_current))
            add2 = sp.csr_matrix((n_current + n_inject, n_inject))
            adj_attack = sp.vstack([adj, add1])
            adj_attack = sp.hstack([adj_attack, add2])
            adj_attack.row = np.hstack([adj_attack.row, new_edges_x])
            adj_attack.col = np.hstack([adj_attack.col, new_edges_y])
            adj_attack.data = np.hstack([adj_attack.data, new_data])

            return adj_attack

        if 'random' in mode:
            for i in range(n_inject):
                islinked = np.zeros(n_test)
                for j in range(n_connect):
                    x = i + n_current

                    yy = random.randint(0, n_test - 1)
                    while islinked[yy] > 0:
                        yy = random.randint(0, n_test - 1)

                    y = test_index[yy]
                    new_edges_x.extend([x, y])
                    new_edges_y.extend([y, x])
                    new_data.extend([1, 1])

            add1 = sp.csr_matrix((n_inject, n_current))
            add2 = sp.csr_matrix((n_current + n_inject, n_inject))
            adj_attack = sp.vstack([adj, add1])
            adj_attack = sp.hstack([adj_attack, add2])
            adj_attack.row = np.hstack([adj_attack.row, new_edges_x])
            adj_attack.col = np.hstack([adj_attack.col, new_edges_y])
            adj_attack.data = np.hstack([adj_attack.data, new_data])

            return adj_attack

        if 'tdgia' in mode:
            add_score = np.zeros(n_test)
            deg = np.array(adj.sum(axis=0))[0] + 1.0
            for i in range(n_test):
                it = test_index[i]
                label = origin_labels[it]
                score = logits[it][label] + 2
                add_score1 = score / deg[it]
                add_score2 = score / np.sqrt(deg[it])
                sc = weight1 * add_score1 + weight2 * add_score2 / np.sqrt(n_connect + n_self_connect)
                add_score[i] = sc

            # higher score is better
            sorted_rank = add_score.argsort()
            sorted_rank = sorted_rank[-n_inject * n_connect:]
            labelgroup = np.zeros(n_classes)

            # separate them by origin_labels
            labelil = []
            for i in range(n_classes):
                labelil.append([])
            random.shuffle(sorted_rank)
            for i in sorted_rank:
                label = origin_labels[test_index[i]]
                labelgroup[label] += 1
                labelil[label].append(i)

            pos = np.zeros(n_classes)
            for i in range(n_inject):
                for j in range(n_connect):
                    smallest = 1
                    smallid = 0
                    for k in range(n_classes):
                        if len(labelil[k]) > 0:
                            if (pos[k] / len(labelil[k])) < smallest:
                                smallest = pos[k] / len(labelil[k])
                                smallid = k

                    tu = labelil[smallid][int(pos[smallid])]

                    pos[smallid] += 1
                    x = n_current + i
                    y = test_index[tu]
                    new_edges_x.extend([x, y])
                    new_edges_y.extend([y, x])
                    new_data.extend([1, 1])

            islinked = np.zeros((n_inject, n_inject))
            for i in range(n_inject):
                rndtimes = 100
                while np.sum(islinked[i]) < n_self_connect and rndtimes > 0:
                    x = i + n_current
                    rndtimes = 100
                    yy = random.randint(0, n_inject - 1)

                    while (np.sum(islinked[yy]) >= n_self_connect or yy == i or
                           islinked[i][yy] == 1) and (rndtimes > 0):
                        yy = random.randint(0, n_inject - 1)
                        rndtimes -= 1

                    if rndtimes > 0:
                        y = n_current + yy
                        islinked[i][yy] = 1
                        islinked[yy][i] = 1
                        new_edges_x.extend([x, y])
                        new_edges_y.extend([y, x])
                        new_data.extend([1, 1])

            add1 = sp.csr_matrix((n_inject, n_current))
            add2 = sp.csr_matrix((n_current + n_inject, n_inject))
            adj_attack = sp.vstack([adj, add1])
            adj_attack = sp.hstack([adj_attack, add2])
            adj_attack.row = np.hstack([adj_attack.row, new_edges_x])
            adj_attack.col = np.hstack([adj_attack.col, new_edges_y])
            adj_attack.data = np.hstack([adj_attack.data, new_data])

            return adj_attack

    def update_features(self, model, adj_attack, features, features_attack, origin_labels,
                        target_mask, adj_norm_func, opt='sin', smooth_factor=4):
        lr = self.lr
        n_epoch = self.n_epoch
        feat_lim_min, feat_lim_max = self.feat_lim_min, self.feat_lim_max

        n_total = features.shape[0]
        adj_attacked_tensor = utils.adj_preprocess(adj=adj_attack,
                                                   adj_norm_func=adj_norm_func,
                                                   model_type=model.model_type,
                                                   device=self.device)

        if opt == 'sin':
            features_attack = features_attack / feat_lim_max
            features_attack = np.arcsin(features_attack)
        features_attack = utils.feat_preprocess(features=features_attack, device=self.device)
        features_attack.requires_grad_(True)
        optimizer = torch.optim.Adam([features_attack], lr=lr)
        loss_func = nn.CrossEntropyLoss(reduction='none')
        model.eval()

        for i in range(n_epoch):
            if opt == 'sin':
                features_attacked = torch.sin(features_attack) * feat_lim_max
            elif opt == 'clip':
                features_attacked = torch.clamp(features_attack, feat_lim_min, feat_lim_max)
            features_concat = torch.cat((features, features_attacked), dim=0)
            pred = model(features_concat, adj_attacked_tensor)
            pred_loss = loss_func(pred[:n_total][target_mask],
                                  origin_labels[target_mask]).to(self.device)
            if opt == 'sin':
                pred_loss = F.relu(-pred_loss + smooth_factor) ** 2
            elif opt == 'clip':
                pred_loss = -pred_loss
            pred_loss = torch.mean(pred_loss)
            optimizer.zero_grad()
            pred_loss.backward(retain_graph=True)
            optimizer.step()
            test_score = metric.eval_acc(pred[:n_total][target_mask],
                                         origin_labels[target_mask])

            if self.early_stop:
                self.early_stop(test_score)
                if self.early_stop.stop:
                    print("Attacking: Early stopped.")
                    return features_attack

            if self.verbose:
                print("Attacking: Epoch {}, Loss: {:.5f}, Surrogate test acc: {:.5f}".format(i, pred_loss, test_score),
                      end='\r' if i != n_epoch - 1 else '\n')

        return features_attacked


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
