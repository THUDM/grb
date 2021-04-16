import os
import pickle
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import grb.utils as utils
from grb.attack.base import InjectionAttack
from grb.utils import evaluator


class TDGIA(InjectionAttack):
    def __init__(self, dataset, n_inject_max, n_edge_max, device='cpu'):
        self.dataset = dataset
        self.adj = dataset.adj
        self.n_total = dataset.num_nodes
        self.n_test = dataset.num_test
        self.n_classes = dataset.num_classes
        self.n_inject_max = n_inject_max
        self.n_edge_max = n_edge_max
        self.test_mask = dataset.test_mask
        self.device = device
        self.config = {}

    def set_config(self, **kwargs):
        self.config['lr'] = kwargs['lr']
        self.config['n_epoch'] = kwargs['n_epoch']
        self.config['feat_lim_min'] = kwargs['feat_lim_min']
        self.config['feat_lim_max'] = kwargs['feat_lim_max']
        self.config['inject_mode'] = kwargs['inject_mode']

    def attack(self, model, features, adj, mode='sequential'):
        """attack mode: sequential, one-time, multi-model"""
        # To-do: adj pre-processing for different models.
        # feature pre-processing.
        model.eval()
        lr = self.config['lr']
        n_epoch = self.config['n_epoch']
        inject_mode = self.config['inject_mode']
        pred_orig = model(features, utils.adj_to_tensor(adj).to(self.device))
        pred_orig_logits = F.softmax(pred_orig, dim=1)
        pred_orig_label = torch.argmax(pred_orig, dim=1)
        feat_lim_min, feat_lim_max = self.config['feat_lim_min'], self.config['feat_lim_max']

        adj_attack = self.injection(label_origin=pred_orig_label,
                                    logits=pred_orig_logits,
                                    n_add=self.n_inject_max,
                                    mode=inject_mode)

        adj_attack = utils.adj_to_tensor(adj_attack).to(self.device)
        features_attack = np.zeros([self.n_inject_max, self.dataset.features.shape[1]])
        features_attack = torch.FloatTensor(features_attack).to(self.device)
        features_attack.requires_grad_(True)
        optimizer = torch.optim.Adam([features_attack], lr=lr)

        for i in range(n_epoch):
            features_concat = torch.cat((features, features_attack), dim=0)
            pred_adv = model(features_concat, adj_attack)
            pred_loss = -F.nll_loss(pred_adv[:self.n_total][self.dataset.test_mask],
                                    pred_orig_label[self.dataset.test_mask]).to(self.device)
            optimizer.zero_grad()
            pred_loss.backward(retain_graph=True)
            optimizer.step()

            # clip
            with torch.no_grad():
                features_attack.clamp_(feat_lim_min, feat_lim_max)
            print("Epoch {}, Loss: {:.5f}, Test acc: {:.5f}".format(i, pred_loss,
                                                                    evaluator.eval_acc(
                                                                        pred_adv[:self.n_total][self.dataset.test_mask],
                                                                        pred_orig_label[self.dataset.test_mask])))

        return adj_attack, features_attack

    def injection(self, label_origin, logits, n_add, self_connect_ratio=0.0, mode='uniform', weight1=0.9, weight2=0.1):
        """Injection mode: uniform, fgsm, random. """
        adj = self.adj
        n_inject = self.n_inject_max
        n_origin = self.n_total
        n_current = n_origin + n_add
        n_connect = int(self.n_edge_max * (1 - self_connect_ratio))
        n_self_connect = int(self.n_edge_max * self_connect_ratio)

        new_edges_x = []
        new_edges_y = []
        new_data = []
        test_index = torch.where(self.test_mask)[0]
        if 'uniform' in mode:
            for i in range(n_inject):
                x = i + n_current
                for j in range(n_connect):
                    id = (x - n_origin) * n_connect + j
                    id = id % self.n_test
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

        if 'fgsm' in mode:
            for i in range(n_inject):
                islinked = np.zeros(self.n_test)
                for j in range(n_connect):
                    x = i + n_current

                    yy = random.randint(0, self.n_test - 1)
                    while islinked[yy] > 0:
                        yy = random.randint(0, self.n_test - 1)

                    y = test_index[yy]
                    new_edges_x.extend([x, y])
                    new_edges_y.extend([y, x])
                    new_data.extend([1, 1])

            add1 = sp.csr_matrix((n_inject, n_current))
            add2 = sp.csr_matrix((n_current + n_inject, n_inject))
            adj_attack = sp.vstack([self.adj, add1])
            adj_attack = sp.hstack([adj_attack, add2])
            adj_attack.row = np.hstack([adj_attack.row, new_edges_x])
            adj_attack.col = np.hstack([adj_attack.col, new_edges_y])
            adj_attack.data = np.hstack([adj_attack.data, new_data])

            return adj_attack

        add_score = np.zeros(self.n_test)
        deg = np.array(self.adj.sum(axis=0))[0] + 1.0
        for i in range(self.n_test):
            it = test_index[i]
            label = label_origin[it]
            score = logits[it][label] + 2
            add_score1 = score / deg[it]
            add_score2 = score / np.sqrt(deg[it])
            sc = weight1 * add_score1 + weight2 * add_score2 / np.sqrt(n_connect + n_self_connect)
            add_score[i] = sc

        # higher score is better
        sorted_rank = add_score.argsort()
        sorted_rank = sorted_rank[-n_connect * n_inject:]
        labelgroup = np.zeros(self.n_classes)

        # separate them by label_origin
        labelil = []
        for i in range(self.n_classes):
            labelil.append([])
        random.shuffle(sorted_rank)
        for i in sorted_rank:
            label = label_origin[test_index[i]]
            labelgroup[label] += 1
            labelil[label].append(i)

        pos = np.zeros(self.n_classes)
        for i in range(n_inject):
            for j in range(n_connect):
                smallest = 1
                smallid = 0
                for k in range(self.n_classes):
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

                while (np.sum(islinked[yy]) >= n_self_connect or yy == i or islinked[i][yy] == 1) and (rndtimes > 0):
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

    def save_features(self, features, file_dir, file_name='features.npy'):
        if not os.path.exists(file_dir):
            assert os.mkdir(file_dir)
        if features is not None:
            np.save(os.path.join(file_dir, file_name), features.detach().numpy())

    def save_adj(self, adj, file_dir, file_name='adj.pkl'):
        if adj is not None:
            with open(os.path.join(file_dir, file_name), 'wb') as f:
                pickle.dump(adj, f)
