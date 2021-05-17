import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append('..')

import grb.utils as utils
from grb.dataset.dataset import CustomDataset
from grb.utils import normalize

sys.path.append('..')


def build_model(model_name, device="cpu"):
    if model_name in "gcn_ln":
        from grb.model.gcn import GCN

        model = GCN(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=[256, 128, 64],
                    layer_norm=True,
                    activation=F.relu)
        adj_norm_func = normalize.GCNAdjNorm
    elif model_name in "graphsage":
        from grb.model.graphsage import GraphSAGE

        model = GraphSAGE(in_features=num_features, out_features=num_classes,
                          hidden_features=[128, 128, 128], activation=F.relu)
        adj_norm_func = normalize.SAGEAdjNorm
    elif model_name in "sgcn":
        from grb.model.sgcn import SGCN

        model = SGCN(in_features=num_features, out_features=num_classes, hidden_features=[128, 128, 128],
                     activation=F.relu)
        adj_norm_func = normalize.GCNAdjNorm
    elif model_name in "robustgcn":
        from grb.model.robustgcn import RobustGCN

        model = RobustGCN(in_features=num_features, out_features=num_classes,
                          hidden_features=[128, 128, 128])
        adj_norm_func = normalize.RobustGCNAdjNorm
    elif model_name in "tagcn":
        from grb.model.tagcn import TAGCN

        model = TAGCN(in_features=num_features, out_features=num_classes, hidden_features=[128, 128, 128],
                      k=2, activation=F.leaky_relu)
        adj_norm_func = normalize.GCNAdjNorm
    elif model_name in "appnp":
        from grb.model.appnp import APPNP

        model = APPNP(in_features=num_features, out_features=num_classes, hidden_features=128,
                      alpha=0.01, k=10)
        adj_norm_func = normalize.GCNAdjNorm
    elif model_name in "gin":
        from grb.model.gin import GIN

        model = GIN(in_features=num_features, out_features=num_classes, hidden_features=[128, 128, 128],
                    activation=F.relu)
        adj_norm_func = normalize.GCNAdjNorm

    model = model.to(device)

    return model, adj_norm_func


def prepare_attack(attack_name, device="cpu", args=None):
    if attack_name in "fgsm":
        from grb.attack.fgsm import FGSM

        config = {}
        config['epsilon'] = args.lr
        config['n_epoch'] = args.n_epoch
        config['feat_lim_min'] = args.feat_lim_min
        config['feat_lim_max'] = args.feat_lim_max
        config['n_inject_max'] = args.n_inject
        config['n_edge_max'] = args.n_edge_max

        attack = FGSM(dataset, adj_norm_func=None, device=device)
        attack.set_config(**config)
    elif attack_name in "speit":
        from grb.attack.speit import SPEIT

        config = {}
        config['inject_mode'] = 'random-inter'
        config['lr'] = args.lr
        config['n_epoch'] = args.n_epoch
        config['feat_lim_min'] = args.feat_lim_min
        config['feat_lim_max'] = args.feat_lim_max
        config['n_inject_max'] = args.n_inject
        config['n_edge_max'] = args.n_edge_max

        attack = SPEIT(dataset, adj_norm_func=None, device=device)
        attack.set_config(**config)

        attack = SPEIT(dataset, adj_norm_func=None, device=device)
        attack.set_config(**config)
    elif attack_name in "tdgia":
        from grb.attack.tdgia import TDGIA

        config = {}
        config['inject_mode'] = 'random'
        config['lr'] = args.lr
        config['n_epoch'] = args.n_epoch
        config['feat_lim_min'] = args.feat_lim_min
        config['feat_lim_max'] = args.feat_lim_max
        config['n_inject_max'] = args.n_inject
        config['n_edge_max'] = args.n_edge_max

        attack = TDGIA(dataset, adj_norm_func=None, device=device)
        attack.set_config(**config)

    return attack


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training GNN models')
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--n_epoch", type=int, default=200, help="Training epoch.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--save_after", type=int, default=200)
    parser.add_argument("--data_dir", type=str, default="/home/qinkai/research/grb/data/Refined-cora-citeseer")
    parser.add_argument("--model_dir", type=str, default="/home/qinkai/research/grb/saved_models/")
    parser.add_argument("--model_list", nargs='+',
                        default=None)  # ["gcn_ln", "graphsage", "sgcn", "robustgcn", "tagcn", "appnp", "gin"]
    parser.add_argument("--model_name", type=str, default="gcn")
    parser.add_argument("--attack_list", nargs='+', default=["fgsm", "speit", "tdgia"])
    parser.add_argument("--attack_name", type=str, default="fgsm")
    parser.add_argument("--save_dir", type=str, default="/home/qinkai/research/grb/results")
    parser.add_argument("--model_suffix", type=str, default="rcora")
    parser.add_argument("--early_stop", action='store_true')
    parser.add_argument("--lr_scheduler", action='store_true')
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_inject", type=int, default=20)
    parser.add_argument("--n_edge_max", type=int, default=20)
    parser.add_argument("--feat_lim_min", type=float, default=-2)
    parser.add_argument("--feat_lim_max", type=float, default=2)

    args = parser.parse_args()

    if args.gpu >= 0:
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    data_dir = args.data_dir
    with open(os.path.join(data_dir, "corax_adj.pkl"), 'rb') as f:
        adj = pickle.load(f)
    with open(os.path.join(data_dir, "corax_features.pkl"), 'rb') as f:
        features = pickle.load(f)
    with open(os.path.join(data_dir, "corax_labels.pkl"), 'rb') as f:
        labels = pickle.load(f)
        labels = np.argmax(labels, axis=1)

    n_node = features.shape[0]
    train_mask = torch.zeros(n_node, dtype=bool)
    train_mask[range(1180)] = True
    val_mask = torch.zeros(n_node, dtype=bool)
    val_mask[range(1180, 2180)] = True
    test_mask = torch.zeros(n_node, dtype=bool)
    test_mask[range(2180, 2680)] = True

    dataset = CustomDataset(adj=adj,
                            features=features,
                            labels=labels,
                            train_mask=train_mask,
                            val_mask=val_mask,
                            test_mask=test_mask,
                            name='Refined-Cora')
    adj = dataset.adj
    adj_tensor = dataset.adj_tensor
    features = dataset.features
    labels = dataset.labels
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    print("Attack vs. Defense...")
    if args.attack_list is not None:
        attack_list = args.attack_list
    else:
        attack_list = [args.attack_name]
    if args.model_list is not None:
        model_list = args.model_list
    else:
        model_list = [args.model_name]

    for attack_name in attack_list:
        attack = prepare_attack(attack_name, device, args)

        for model_name in model_list:
            print("{} vs. {}...".format(attack_name, model_name))
            model, adj_norm_func = build_model(model_name, device)
            attack.adj_norm_func = adj_norm_func
            if attack_name in "speit":
                target_node = np.random.choice(dataset.num_test, 1000)
                adj_attack, features_attack = attack.attack(model, target_node)
            else:
                adj_attack, features_attack = attack.attack(model)
            save_dir = os.path.join(args.save_dir, attack_name + "_vs_" + model_name)
            utils.save_adj(adj_attack.tocsr()[-args.n_inject:, :], save_dir)
            utils.save_features(features_attack, save_dir)
    print("Attack finished.")
