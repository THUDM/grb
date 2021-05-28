import torch
import torch.nn.functional as F

import grb.utils as utils
from grb.evaluator import metric

model_list = ["gcn", "gcn_ln", "graphsage", "sgcn",
              "robustgcn", "tagcn", "appnp", "gin"]

attack_list = ["rnd", "fgsm", "pgd", "speit", "tdgia"]


def build_model(model_name, num_features, num_classes):
    if model_name in "gcn":
        from grb.model.torch.gcn import GCN

        model = GCN(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=[128, 128, 128],
                    activation=F.relu)
        adj_norm_func = utils.normalize.GCNAdjNorm
    elif model_name in "gcn_ln":
        from grb.model.torch.gcn import GCN

        model = GCN(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=[128, 128, 128],
                    layer_norm=True,
                    activation=F.relu)
        adj_norm_func = utils.normalize.GCNAdjNorm
    elif model_name in "graphsage":
        from grb.model.torch.graphsage import GraphSAGE

        model = GraphSAGE(in_features=num_features,
                          out_features=num_classes,
                          hidden_features=[128, 128, 128],
                          activation=F.relu)
        adj_norm_func = utils.normalize.SAGEAdjNorm
    elif model_name in "sgcn":
        from grb.model.torch.sgcn import SGCN

        model = SGCN(in_features=num_features,
                     out_features=num_classes,
                     hidden_features=[128, 128, 128],
                     activation=F.relu)
        adj_norm_func = utils.normalize.GCNAdjNorm
    elif model_name in "robustgcn":
        from grb.model.torch.robustgcn import RobustGCN

        model = RobustGCN(in_features=num_features,
                          out_features=num_classes,
                          hidden_features=[128, 128, 128])
        adj_norm_func = utils.normalize.RobustGCNAdjNorm
    elif model_name in "tagcn":
        from grb.model.torch.tagcn import TAGCN

        model = TAGCN(in_features=num_features,
                      out_features=num_classes,
                      hidden_features=[128, 128, 128],
                      k=2, activation=F.leaky_relu)
        adj_norm_func = utils.normalize.GCNAdjNorm
    elif model_name in "appnp":
        from grb.model.torch.appnp import APPNP

        model = APPNP(in_features=num_features,
                      out_features=num_classes,
                      hidden_features=128,
                      alpha=0.01, k=10)
        adj_norm_func = utils.normalize.GCNAdjNorm
    elif model_name in "gin":
        from grb.model.torch.gin import GIN

        model = GIN(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=[128, 128, 128],
                    activation=F.relu)
        adj_norm_func = utils.normalize.GCNAdjNorm

    return model, adj_norm_func


def build_optimizer(model, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return optimizer


def build_loss():
    return F.binary_cross_entropy


def build_metric():
    return metric.eval_f1multilabel


def build_attack(attack_name, device="cpu", args=None):
    if attack_name in "rnd":
        from grb.attack.rnd import RND

        attack = RND(n_inject_max=args.n_inject,
                     n_edge_max=args.n_edge_max,
                     feat_lim_min=args.feat_lim_min,
                     feat_lim_max=args.feat_lim_max,
                     device=device)
    elif attack_name in "fgsm":
        from grb.attack.fgsm import FGSM

        attack = FGSM(epsilon=args.lr,
                      n_epoch=args.n_epoch,
                      n_inject_max=args.n_inject,
                      n_edge_max=args.n_edge_max,
                      feat_lim_min=args.feat_lim_min,
                      feat_lim_max=args.feat_lim_max,
                      device=device)
    elif attack_name in "pgd":
        from grb.attack.pgd import PGD

        attack = PGD(epsilon=args.lr,
                     n_epoch=args.n_epoch,
                     n_inject_max=args.n_inject,
                     n_edge_max=args.n_edge_max,
                     feat_lim_min=args.feat_lim_min,
                     feat_lim_max=args.feat_lim_max,
                     device=device)
    elif attack_name in "speit":
        from grb.attack.speit import SPEIT

        attack = SPEIT(lr=args.lr,
                       n_epoch=args.n_epoch,
                       n_inject_max=args.n_inject,
                       n_edge_max=args.n_edge_max,
                       feat_lim_min=args.feat_lim_min,
                       feat_lim_max=args.feat_lim_max,
                       device=device)
    elif attack_name in "tdgia":
        from grb.attack.tdgia import TDGIA

        attack = TDGIA(lr=args.lr,
                       n_epoch=args.n_epoch,
                       n_inject_max=args.n_inject,
                       n_edge_max=args.n_edge_max,
                       feat_lim_min=args.feat_lim_min,
                       feat_lim_max=args.feat_lim_max,
                       device=device)

    return attack
