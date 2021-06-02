import torch
import torch.nn.functional as F

import grb.utils as utils
from grb.evaluator import metric

model_list = ["gcn",
              "gcn_ln",
              "graphsage",
              "graphsage_ln",
              "sgcn",
              "sgcn_ln",
              "robustgcn",
              "tagcn",
              "tagcn_ln",
              "appnp",
              "appnp_ln",
              "gin",
              "gin_ln",
              "gat",
              "gat_ln"]

attack_list = ["rnd", "fgsm", "pgd", "speit", "tdgia"]

model_sur_list = ["gcn"]


def build_model(model_name, num_features, num_classes, layer_norm=False):
    if model_name in "gcn":
        from grb.model.torch.gcn import GCN

        model = GCN(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=[64, 64, 64],
                    activation=F.relu)
        adj_norm_func = utils.normalize.GCNAdjNorm

    elif model_name in "gcn_ln":
        from grb.model.torch.gcn import GCN

        model = GCN(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=[64, 64, 64],
                    layer_norm=True,
                    activation=F.relu)
        adj_norm_func = utils.normalize.GCNAdjNorm

    elif model_name in "graphsage":
        from grb.model.torch.graphsage import GraphSAGE

        model = GraphSAGE(in_features=num_features,
                          out_features=num_classes,
                          hidden_features=[64, 64, 64],
                          activation=F.relu)
        adj_norm_func = utils.normalize.SAGEAdjNorm

    elif model_name in "graphsage_ln":
        from grb.model.torch.graphsage import GraphSAGE

        model = GraphSAGE(in_features=num_features,
                          out_features=num_classes,
                          hidden_features=[64, 64, 64],
                          layer_norm=True,
                          activation=F.relu)
        adj_norm_func = utils.normalize.SAGEAdjNorm

    elif model_name in "sgcn":
        from grb.model.torch.sgcn import SGCN

        model = SGCN(in_features=num_features,
                     out_features=num_classes,
                     hidden_features=[64, 64, 64],
                     activation=F.relu)
        adj_norm_func = utils.normalize.GCNAdjNorm

    elif model_name in "sgcn_ln":
        from grb.model.torch.sgcn import SGCN

        model = SGCN(in_features=num_features,
                     out_features=num_classes,
                     hidden_features=[64, 64, 64],
                     layer_norm=True,
                     activation=F.relu)
        adj_norm_func = utils.normalize.GCNAdjNorm

    elif model_name in "robustgcn":
        from grb.model.torch.robustgcn import RobustGCN

        model = RobustGCN(in_features=num_features,
                          out_features=num_classes,
                          hidden_features=[64, 64, 64])
        adj_norm_func = utils.normalize.RobustGCNAdjNorm

    elif model_name in "tagcn":
        from grb.model.torch.tagcn import TAGCN

        model = TAGCN(in_features=num_features,
                      out_features=num_classes,
                      hidden_features=[64, 64, 64],
                      k=2, activation=F.leaky_relu)
        adj_norm_func = utils.normalize.GCNAdjNorm

    elif model_name in "tagcn_ln":
        from grb.model.torch.tagcn import TAGCN

        model = TAGCN(in_features=num_features,
                      out_features=num_classes,
                      hidden_features=[64, 64, 64],
                      layer_norm=True,
                      k=2, activation=F.leaky_relu)
        adj_norm_func = utils.normalize.GCNAdjNorm

    elif model_name in "appnp":
        from grb.model.torch.appnp import APPNP

        model = APPNP(in_features=num_features,
                      out_features=num_classes,
                      hidden_features=64,
                      alpha=0.01, k=10)
        adj_norm_func = utils.normalize.GCNAdjNorm

    elif model_name in "appnp_ln":
        from grb.model.torch.appnp import APPNP

        model = APPNP(in_features=num_features,
                      out_features=num_classes,
                      hidden_features=64,
                      layer_norm=True,
                      alpha=0.01, k=10)
        adj_norm_func = utils.normalize.GCNAdjNorm

    elif model_name in "gin":
        from grb.model.torch.gin import GIN

        model = GIN(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=[64, 64, 64],
                    activation=F.relu)
        adj_norm_func = utils.normalize.GCNAdjNorm

    elif model_name in "gin_ln":
        from grb.model.torch.gin import GIN

        model = GIN(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=[64, 64, 64],
                    layer_norm=True,
                    activation=F.relu)
        adj_norm_func = utils.normalize.GCNAdjNorm

    elif model_name in "gat":
        from grb.model.dgl.gat import GAT

        model = GAT(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=[64, 64, 64],
                    num_heads=4,
                    layer_norm=False,
                    activation=F.leaky_relu)
        adj_norm_func = utils.normalize.GCNAdjNorm

    elif model_name in "gat_ln":
        from grb.model.dgl.gat import GAT

        model = GAT(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=[64, 64, 64],
                    num_heads=4,
                    layer_norm=True,
                    activation=F.leaky_relu)
        adj_norm_func = utils.normalize.GCNAdjNorm
    elif model_name in "gcnguard":
        from grb.defense.gnnguard import GCNGuard

        model = GCNGuard(in_features=num_features,
                         out_features=num_classes,
                         hidden_features=[64, 64],
                         activation=F.relu,
                         drop=True)
        adj_norm_func = utils.normalize.GCNAdjNorm
    elif model_name in "gatguard":
        from grb.defense.gnnguard import GATGuard

        model = GATGuard(in_features=num_features,
                         out_features=num_classes,
                         hidden_features=[64, 64],
                         num_heads=4,
                         activation=F.relu,
                         drop=True)
        adj_norm_func = utils.normalize.GCNAdjNorm
    else:
        raise NotImplementedError

    return model, adj_norm_func


def build_optimizer(model, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return optimizer


def build_loss():
    return F.nll_loss


def build_metric():
    return metric.eval_acc


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
                      early_stop=args.early_stop,
                      device=device)
    elif attack_name in "pgd":
        from grb.attack.pgd import PGD

        attack = PGD(epsilon=args.lr,
                     n_epoch=args.n_epoch,
                     n_inject_max=args.n_inject,
                     n_edge_max=args.n_edge_max,
                     feat_lim_min=args.feat_lim_min,
                     feat_lim_max=args.feat_lim_max,
                     early_stop=args.early_stop,
                     device=device)
    elif attack_name in "speit":
        from grb.attack.speit import SPEIT

        attack = SPEIT(lr=args.lr,
                       n_epoch=args.n_epoch,
                       n_inject_max=args.n_inject,
                       n_edge_max=args.n_edge_max,
                       feat_lim_min=args.feat_lim_min,
                       feat_lim_max=args.feat_lim_max,
                       early_stop=args.early_stop,
                       device=device)
    elif attack_name in "tdgia":
        from grb.attack.tdgia import TDGIA

        attack = TDGIA(lr=args.lr,
                       n_epoch=args.n_epoch,
                       n_inject_max=args.n_inject,
                       n_edge_max=args.n_edge_max,
                       feat_lim_min=args.feat_lim_min,
                       feat_lim_max=args.feat_lim_max,
                       early_stop=args.early_stop,
                       inject_mode='random',
                       sequential_step=1.0,
                       device=device)
    elif attack_name in "tdgia_random":
        from grb.attack.tdgia import TDGIA

        attack = TDGIA(lr=args.lr,
                       n_epoch=args.n_epoch,
                       n_inject_max=args.n_inject,
                       n_edge_max=args.n_edge_max,
                       feat_lim_min=args.feat_lim_min,
                       feat_lim_max=args.feat_lim_max,
                       early_stop=args.early_stop,
                       inject_mode='random',
                       device=device)
    elif attack_name in "tdgia_uniform":
        from grb.attack.tdgia import TDGIA

        attack = TDGIA(lr=args.lr,
                       n_epoch=args.n_epoch,
                       n_inject_max=args.n_inject,
                       n_edge_max=args.n_edge_max,
                       feat_lim_min=args.feat_lim_min,
                       feat_lim_max=args.feat_lim_max,
                       early_stop=args.early_stop,
                       inject_mode='uniform',
                       sequential_step=1.0,
                       device=device)
    else:
        raise NotImplementedError

    return attack
