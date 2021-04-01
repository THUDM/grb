import os
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from grb.attack.base import InjectionAttack
from grb.utils import utils, evaluator


class Speit(InjectionAttack):
    def __init__(self, dataset, n_epoch, n_inject, n_edge_max, device='cpu'):
        self.dataset = dataset
        self.n_epoch = n_epoch
        self.n_test = np.sum(dataset.test_mask)
        self.n_total = dataset.adj.shape[0]
        self.n_inject = n_inject
        self.n_edge_max = n_edge_max
        self.device = device
        self.config = {}

    def set_config(self, **kwargs):
        self.config['lr'] = kwargs['lr']
        self.config['feat_lim_min'] = kwargs['feat_lim_min']
        self.config['feat_lim_max'] = kwargs['feat_lim_max']
        self.config['mode'] = kwargs['mode']

    def attack(self, model, features, adj, target_node):
        model.eval()
        lr = self.config['lr']
        pred_orig = model(features, utils.adj_to_tensor(adj).to(self.device))
        pred_orig_label = torch.argmax(pred_orig, dim=1)
        feat_lim_min, feat_lim_max = self.config['feat_lim_min'], self.config['feat_lim_max']

        adj_attack = self.injection(target_node, self.config['mode'])
        adj_attack = utils.adj_to_tensor(adj_attack).to(self.device)
        features_adv = np.zeros([self.n_inject, self.dataset.features.shape[1]])
        features_adv = torch.FloatTensor(features_adv).to(self.device)
        features_adv.requires_grad_(True)
        optimizer = torch.optim.Adam([features_adv], lr=lr)

        for i in range(self.n_epoch):
            features_concat = torch.cat((features, features_adv), dim=0)
            pred_adv = model(features_concat, adj_attack)
            pred_loss = -F.nll_loss(pred_adv[:self.n_total][self.dataset.test_mask],
                                    pred_orig_label[self.dataset.test_mask]).to(self.device)
            optimizer.zero_grad()
            pred_loss.backward(retain_graph=True)
            optimizer.step()

            # clip
            with torch.no_grad():
                features_adv.clamp_(feat_lim_min, feat_lim_max)
            print("Epoch {}, Loss: {:.5f}, Test acc: {:.5f}".format(i, pred_loss,
                                                                    evaluator.eval_acc(
                                                                        pred_adv[:self.n_total][self.dataset.test_mask],
                                                                        pred_orig_label[self.dataset.test_mask])))

        return features_adv

    def injection(self, target_node, mode='random-inter'):

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
                print("Mode ERROR: 'mode' should be one of ['random-inter', 'multi-layer']")

            return adj

        # construct injected adjacency matrix
        adj_inject = inject(target_node, self.n_inject, self.n_test, mode)
        adj_inject = np.concatenate([np.zeros([self.n_inject, self.n_total - self.n_test]), adj_inject], axis=1)
        adj_inject = sp.csr_matrix(adj_inject)
        adj_attack = sp.vstack([self.dataset.adj, adj_inject[:, :self.n_total]])
        adj_attack = sp.hstack([adj_attack, adj_inject.T])
        adj_attack = sp.coo_matrix(adj_attack)

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
