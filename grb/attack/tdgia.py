import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import InjectionAttack, EarlyStop
from ..evaluator import metric
from ..utils import utils


class TDGIA(InjectionAttack):
    r"""
    Description
    -----------
    Topological Defective Graph Injection Attack (`TDGIA <https://github.com/THUDM/tdgia>`__).

    Parameters
    ----------
    lr : float
        Learning rate of feature optimization process.
    n_epoch : int
        Epoch of perturbations.
    n_inject_max : int
        Maximum number of injected nodes.
    n_edge_max : int
        Maximum number of edges of injected nodes.
    feat_lim_min : float
        Minimum limit of features.
    feat_lim_max : float
        Maximum limit of features.
    loss : func of torch.nn.functional, optional
        Loss function compatible with ``torch.nn.functional``. Default: ``F.nll_loss``.
    eval_metric : func of grb.evaluator.metric, optional
        Evaluation metric. Default: ``metric.eval_acc``.
    inject_mode : str, optional
        Mode of injection. Choose from ``["random", "uniform", "tdgia"]``. Default: ``tdgia``.
    sequential_step : float, optional
        Step of sequential injection, each time injecting :math:`\alpha\times N_{inject}` nodes. Default: ``0.2``.
    opt : str, optional
        Optimization option. Choose from ``["sin", "clip"]``. Default: ``sin``.
    device : str, optional
        Device used to host data. Default: ``cpu``.
    early_stop : bool, optional
        Whether to early stop. Default: ``False``.
    verbose : bool, optional
        Whether to display logs. Default: ``True``.

    """
    def __init__(self,
                 lr,
                 n_epoch,
                 n_inject_max,
                 n_edge_max,
                 feat_lim_min,
                 feat_lim_max,
                 loss=F.nll_loss,
                 eval_metric=metric.eval_acc,
                 inject_mode='tdgia',
                 sequential_step=0.2,
                 opt='sin',
                 device='cpu',
                 early_stop=False,
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
        self.sequential_step = sequential_step
        self.opt = opt
        self.verbose = verbose

        # Early stop
        if early_stop:
            self.early_stop = EarlyStop(patience=1000, epsilon=1e-4)
        else:
            self.early_stop = early_stop

    def attack(self, model, adj, features, target_mask, adj_norm_func):
        r"""

        Description
        -----------
        Attack process consists of injection and feature update.

        Parameters
        ----------
        model : torch.nn.module
            Model implemented based on ``torch.nn.module``.
        adj : scipy.sparse.csr.csr_matrix
            Adjacency matrix in form of ``N * N`` sparse matrix.
        features : torch.FloatTensor
            Features in form of ``N * D`` torch float tensor.
        target_mask : torch.Tensor
            Mask of attack target nodes in form of ``N * 1`` torch bool tensor.
        adj_norm_func : func of utils.normalize
            Function that normalizes adjacency matrix.

        Returns
        -------
        adj_attack : scipy.sparse.csr.csr_matrix
            Adversarial adjacency matrix in form of :math:`(N + N_{inject})\times(N + N_{inject})` sparse matrix.
        features_attack : torch.FloatTensor
            Features of nodes after attacks in form of :math:`N_{inject}` * D` torch float tensor.

        """

        model.to(self.device)
        n_total, n_feat = features.shape
        features = utils.feat_preprocess(features=features, device=self.device)
        adj_tensor = utils.adj_preprocess(adj=adj,
                                          adj_norm_func=adj_norm_func,
                                          model_type=model.model_type,
                                          device=self.device)
        pred_orig = model(features, adj_tensor)
        origin_labels = torch.argmax(pred_orig, dim=1)

        n_inject = 0
        features_attack = features
        """Sequential injection"""
        while n_inject < self.n_inject_max:
            with torch.no_grad():
                adj_tensor = utils.adj_preprocess(adj=adj,
                                                  adj_norm_func=adj_norm_func,
                                                  model_type=model.model_type,
                                                  device=self.device)
                current_labels = F.softmax(model(features_attack, adj_tensor), dim=1)
            n_inject_cur = self.n_inject_max - n_inject
            if n_inject_cur > self.n_inject_max * self.sequential_step:
                n_inject_cur = int(self.n_inject_max * self.sequential_step)

            print("Attacking: Sequential inject {}/{} nodes".format(n_inject + n_inject_cur, self.n_inject_max))
            adj_attack = self.injection(adj=adj,
                                        n_inject=n_inject_cur,
                                        n_origin=n_total,
                                        n_current=n_total + n_inject,
                                        origin_labels=origin_labels,
                                        current_labels=current_labels,
                                        target_mask=target_mask,
                                        mode=self.inject_mode)
            if n_inject < self.n_inject_max:
                n_inject += n_inject_cur
                adj = adj_attack
                features_attack_add = torch.randn((n_inject_cur, n_feat)).to(self.device)
                features_attack_add = self.update_features(model=model,
                                                           adj_attack=adj_attack,
                                                           n_origin=n_total,
                                                           features_current=features_attack,
                                                           features_attack=features_attack_add,
                                                           origin_labels=origin_labels,
                                                           target_mask=target_mask,
                                                           adj_norm_func=adj_norm_func,
                                                           opt=self.opt)
                features_attack = torch.cat((features, features_attack_add), dim=0)

        features_attack = features_attack[n_total:]

        return adj_attack, features_attack

    def injection(self, adj, n_inject, n_origin, n_current,
                  origin_labels, current_labels, target_mask,
                  self_connect_ratio=0.0, weight1=0.9, weight2=0.1, mode='tdgia'):
        r"""

        Description
        -----------
        Randomly inject nodes to target nodes.

        Parameters
        ----------
        adj : scipy.sparse.csr.csr_matrix
            Adjacency matrix in form of ``N * N`` sparse matrix.
        n_inject : int
            Number of injection.
        n_origin : int
            Number of original nodes.
        n_current : int
            Number of current nodes (after injection).
        target_mask : torch.Tensor
            Mask of attack target nodes in form of ``N * 1`` torch bool tensor.
        self_connect_ratio : float. optional
            Ratio of self connected edges among injected nodes. Default: ``0.0``.
        weight1 : float, optional
            Hyper-parameter of the score function. Refer to the paper. Default: ``0.9``.
        weight2 : float, optional
            Hyper-parameter of the score function. Refer to the paper. Default: ``0.1``.
        mode : str, optional
            Mode of injection. Choose from ``["random", "uniform", "tdgia"]``. Default: ``tdgia``.

        Returns
        -------
        adj_attack : scipy.sparse.csr.csr_matrix
            Adversarial adjacency matrix in form of :math:`(N + N_{inject})\times(N + N_{inject})` sparse matrix.

        """

        n_origin = origin_labels.shape[0]
        n_test = torch.sum(target_mask).item()
        n_classes = origin_labels.max() + 1
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
                is_linked = np.zeros(n_test)
                for j in range(n_connect):
                    x = i + n_current

                    yy = random.randint(0, n_test - 1)
                    while is_linked[yy] > 0:
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
                score = current_labels[it][label] + 2
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
                    small_id = 0
                    for k in range(n_classes):
                        if len(labelil[k]) > 0:
                            if (pos[k] / len(labelil[k])) < smallest:
                                smallest = pos[k] / len(labelil[k])
                                small_id = k

                    tu = labelil[small_id][int(pos[small_id])]

                    pos[small_id] += 1
                    x = n_current + i
                    y = test_index[tu]
                    new_edges_x.extend([x, y])
                    new_edges_y.extend([y, x])
                    new_data.extend([1, 1])

            is_linked = np.zeros((n_inject, n_inject))
            for i in range(n_inject):
                rnd_times = 100
                while np.sum(is_linked[i]) < n_self_connect and rnd_times > 0:
                    x = i + n_current
                    rnd_times = 100
                    yy = random.randint(0, n_inject - 1)

                    while (np.sum(is_linked[yy]) >= n_self_connect or yy == i or
                           is_linked[i][yy] == 1) and (rnd_times > 0):
                        yy = random.randint(0, n_inject - 1)
                        rnd_times -= 1

                    if rnd_times > 0:
                        y = n_current + yy
                        is_linked[i][yy] = 1
                        is_linked[yy][i] = 1
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

    def update_features(self, model, adj_attack, features_current, features_attack,
                        origin_labels, n_origin, target_mask, adj_norm_func, opt='sin', smooth_factor=4):
        r"""
        Description
        -----------
        Adversarial feature generation of injected nodes.

        Parameters
        ----------
        model : torch.nn.module
            Model implemented based on ``torch.nn.module``.
        adj_attack :  scipy.sparse.csr.csr_matrix
            Adversarial adjacency matrix in form of :math:`(N + N_{inject})\times(N + N_{inject})` sparse matrix.
        features_current : torch.FloatTensor
            Current features in form of :math:`(N + N_{inject})` * D` torch float tensor.
        features_attack : torch.FloatTensor
            Features of nodes after attacks in form of :math:`N_{inject}` * D` torch float tensor.
        origin_labels : torch.LongTensor
            Labels of target nodes originally predicted by the model.
        n_origin : int
            Number of original nodes.
        target_mask : torch.Tensor
            Mask of target nodes in form of ``N * 1`` torch bool tensor.
        adj_norm_func : func of utils.normalize
            Function that normalizes adjacency matrix.
        opt : str, optional
            Optimization option. Choose from ``["sin", "clip"]``. Default: ``sin``.
        smooth_factor : float, optional
            Factor for smoothing the optimization. Default: ``4``.

        Returns
        -------
        features_attack : torch.FloatTensor
            Updated features of nodes after attacks in form of :math:`N_{inject}` * D` torch float tensor.

        """

        lr = self.lr
        n_epoch = self.n_epoch
        feat_lim_min, feat_lim_max = self.feat_lim_min, self.feat_lim_max

        features_origin = features_current[:n_origin]
        features_added = features_current[n_origin:].cpu().data.numpy()
        if opt == 'sin':
            features_added = features_added / feat_lim_max
            features_added = np.arcsin(features_added)
        features_added = utils.feat_preprocess(features=features_added, device=self.device)

        adj_attacked_tensor = utils.adj_preprocess(adj=adj_attack,
                                                   adj_norm_func=adj_norm_func,
                                                   model_type=model.model_type,
                                                   device=self.device)

        features_attack = utils.feat_preprocess(features=features_attack, device=self.device)
        features_attack.requires_grad_(True)
        optimizer = torch.optim.Adam([features_added, features_attack], lr=lr)
        loss_func = nn.CrossEntropyLoss(reduction='none')
        model.eval()

        for i in range(n_epoch):
            if opt == 'sin':
                features_add = torch.sin(features_added) * feat_lim_max
                features_attacked = torch.sin(features_attack) * feat_lim_max
            elif opt == 'clip':
                features_add = torch.clamp(features_added, feat_lim_min, feat_lim_max)
                features_attacked = torch.clamp(features_attack, feat_lim_min, feat_lim_max)
            features_concat = torch.cat((features_origin, features_add, features_attacked), dim=0)
            pred = model(features_concat, adj_attacked_tensor)
            pred_loss = loss_func(pred[:n_origin][target_mask],
                                  origin_labels[target_mask]).to(self.device)
            if opt == 'sin':
                pred_loss = F.relu(-pred_loss + smooth_factor) ** 2
            elif opt == 'clip':
                pred_loss = -pred_loss
            pred_loss = torch.mean(pred_loss)
            optimizer.zero_grad()
            pred_loss.backward(retain_graph=True)
            optimizer.step()
            test_score = metric.eval_acc(pred[:n_origin][target_mask],
                                         origin_labels[target_mask])

            if self.early_stop:
                self.early_stop(test_score)
                if self.early_stop.stop:
                    print("Attacking: Early stopped.")
                    self.early_stop = EarlyStop()
                    return features_concat[n_origin:].detach()

            if self.verbose:
                print("Attacking: Epoch {}, Loss: {:.5f}, Surrogate test acc: {:.5f}".format(i, pred_loss, test_score),
                      end='\r' if i != n_epoch - 1 else '\n')

        return features_concat[n_origin:].detach()
