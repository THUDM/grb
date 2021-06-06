import torch
import torch.nn.functional as F

import grb.utils as utils
from grb.evaluator import metric

model_list = ["gcn",
              "gcn_ln",
              "gcn_at",
              "graphsage",
              "graphsage_ln",
              "graphsage_at",
              "sgcn",
              "sgcn_at",
              "sgcn_ln",
              "robustgcn",
              "robustgcn_at",
              "tagcn",
              "tagcn_ln",
              "tagcn_at",
              "appnp",
              "appnp_ln",
              "appnp_at",
              "gin",
              "gin_ln",
              "gin_at",
              "gat",
              "gat_ln",
              "gat_at"]

attack_list = ["rnd", "fgsm", "pgd", "speit", "tdgia"]

model_sur_list = ["gcn"]


def build_model(model_name, num_features, num_classes):
    if "guard" in model_name:
        if "gcn" in model_name:
            from grb.defense.gnnguard import GCNGuard

            model = GCNGuard(in_features=num_features,
                             out_features=num_classes,
                             hidden_features=[64, 64],
                             activation=F.relu,
                             drop=True)
            adj_norm_func = utils.normalize.GCNAdjNorm
        elif "gat" in model_name:
            from grb.defense.gnnguard import GATGuard

            model = GATGuard(in_features=num_features,
                             out_features=num_classes,
                             hidden_features=[64, 64],
                             num_heads=2,
                             activation=F.relu,
                             drop=True)
            adj_norm_func = utils.normalize.GCNAdjNorm
    elif "svd" in model_name:
        if "gcn" in model_name:
            from grb.defense.gcnsvd import GCNSVD

            model = GCNSVD(in_features=num_features,
                           out_features=num_classes,
                           hidden_features=[128, 128, 128],
                           activation=F.relu)
            adj_norm_func = utils.normalize.GCNAdjNorm
    elif "robustgcn" in model_name:
        from grb.model.torch.robustgcn import RobustGCN

        model = RobustGCN(in_features=num_features,
                          out_features=num_classes,
                          hidden_features=[128, 128, 128])
        adj_norm_func = utils.normalize.RobustGCNAdjNorm
    elif "sgcn" in model_name:
        from grb.model.torch.sgcn import SGCN
        adj_norm_func = utils.normalize.GCNAdjNorm
        if 'ln' in model_name:
            model = SGCN(in_features=num_features,
                         out_features=num_classes,
                         hidden_features=[128, 128, 128],
                         layer_norm=True,
                         activation=F.relu)
        else:
            model = SGCN(in_features=num_features,
                         out_features=num_classes,
                         hidden_features=[128, 128, 128],
                         activation=F.relu)
    elif "tagcn" in model_name:
        from grb.model.torch.tagcn import TAGCN
        adj_norm_func = utils.normalize.GCNAdjNorm
        if 'ln' in model_name:
            model = TAGCN(in_features=num_features,
                          out_features=num_classes,
                          hidden_features=[128, 128, 128],
                          layer_norm=True,
                          k=2, activation=F.leaky_relu)
        else:
            model = TAGCN(in_features=num_features,
                          out_features=num_classes,
                          hidden_features=[128, 128, 128],
                          k=2, activation=F.leaky_relu)
    elif "gcn" in model_name:
        from grb.model.torch.gcn import GCN
        adj_norm_func = utils.normalize.GCNAdjNorm
        if 'ln' in model_name:
            model = GCN(in_features=num_features,
                        out_features=num_classes,
                        hidden_features=[128, 128, 128],
                        layer_norm=True,
                        activation=F.relu)
        else:
            model = GCN(in_features=num_features,
                        out_features=num_classes,
                        hidden_features=[128, 128, 128],
                        activation=F.relu)
    elif "graphsage" in model_name:
        from grb.model.torch.graphsage import GraphSAGE
        adj_norm_func = utils.normalize.SAGEAdjNorm
        if 'ln' in model_name:
            model = GraphSAGE(in_features=num_features,
                              out_features=num_classes,
                              hidden_features=[128, 128, 128],
                              layer_norm=True,
                              activation=F.relu)
        else:
            model = GraphSAGE(in_features=num_features,
                              out_features=num_classes,
                              hidden_features=[128, 128, 128],
                              activation=F.relu)
    elif "appnp" in model_name:
        from grb.model.torch.appnp import APPNP
        adj_norm_func = utils.normalize.GCNAdjNorm
        if 'ln' in model_name:
            model = APPNP(in_features=num_features,
                          out_features=num_classes,
                          hidden_features=128,
                          layer_norm=True,
                          alpha=0.01, k=10)
        else:
            model = APPNP(in_features=num_features,
                          out_features=num_classes,
                          hidden_features=128,
                          alpha=0.01, k=10)
    elif "gin" in model_name:
        from grb.model.torch.gin import GIN
        adj_norm_func = utils.normalize.GCNAdjNorm
        if 'ln' in model_name:
            model = GIN(in_features=num_features,
                        out_features=num_classes,
                        hidden_features=[128, 128, 128],
                        layer_norm=True,
                        activation=F.relu)
        else:
            model = GIN(in_features=num_features,
                        out_features=num_classes,
                        hidden_features=[128, 128, 128],
                        activation=F.relu)
    elif "gat" in model_name:
        from grb.model.dgl.gat import GAT
        adj_norm_func = utils.normalize.GCNAdjNorm
        if 'ln' in model_name:
            model = GAT(in_features=num_features,
                        out_features=num_classes,
                        hidden_features=[64, 64, 64],
                        num_heads=4,
                        layer_norm=True,
                        activation=F.leaky_relu)
        else:
            model = GAT(in_features=num_features,
                        out_features=num_classes,
                        hidden_features=[64, 64, 64],
                        num_heads=4,
                        layer_norm=False,
                        activation=F.leaky_relu)
    # if model_name in "gcn":
    #     from grb.model.torch.gcn import GCN
    #
    #     model = GCN(in_features=num_features,
    #                 out_features=num_classes,
    #                 hidden_features=[128, 128, 128],
    #                 activation=F.relu)
    #     adj_norm_func = utils.normalize.GCNAdjNorm
    #
    # elif model_name in "gcn_ln":
    #     from grb.model.torch.gcn import GCN
    #
    #     model = GCN(in_features=num_features,
    #                 out_features=num_classes,
    #                 hidden_features=[128, 128, 128],
    #                 layer_norm=True,
    #                 activation=F.relu)
    #     adj_norm_func = utils.normalize.GCNAdjNorm
    #
    # elif model_name in "graphsage":
    #     from grb.model.torch.graphsage import GraphSAGE
    #
    #     model = GraphSAGE(in_features=num_features,
    #                       out_features=num_classes,
    #                       hidden_features=[128, 128, 128],
    #                       activation=F.relu)
    #     adj_norm_func = utils.normalize.SAGEAdjNorm
    #
    # elif model_name in "graphsage_ln":
    #     from grb.model.torch.graphsage import GraphSAGE
    #
    #     model = GraphSAGE(in_features=num_features,
    #                       out_features=num_classes,
    #                       hidden_features=[128, 128, 128],
    #                       layer_norm=True,
    #                       activation=F.relu)
    #     adj_norm_func = utils.normalize.SAGEAdjNorm
    #
    # elif model_name in "sgcn":
    #     from grb.model.torch.sgcn import SGCN
    #
    #     model = SGCN(in_features=num_features,
    #                  out_features=num_classes,
    #                  hidden_features=[128, 128, 128],
    #                  activation=F.relu)
    #     adj_norm_func = utils.normalize.GCNAdjNorm
    #
    # elif model_name in "sgcn_ln":
    #     from grb.model.torch.sgcn import SGCN
    #
    #     model = SGCN(in_features=num_features,
    #                  out_features=num_classes,
    #                  hidden_features=[128, 128, 128],
    #                  layer_norm=True,
    #                  activation=F.relu)
    #     adj_norm_func = utils.normalize.GCNAdjNorm
    #
    # elif model_name in "robustgcn":
    #     from grb.model.torch.robustgcn import RobustGCN
    #
    #     model = RobustGCN(in_features=num_features,
    #                       out_features=num_classes,
    #                       hidden_features=[128, 128, 128])
    #     adj_norm_func = utils.normalize.RobustGCNAdjNorm
    #
    # elif model_name in "tagcn":
    #     from grb.model.torch.tagcn import TAGCN
    #
    #     model = TAGCN(in_features=num_features,
    #                   out_features=num_classes,
    #                   hidden_features=[128, 128, 128],
    #                   k=2, activation=F.leaky_relu)
    #     adj_norm_func = utils.normalize.GCNAdjNorm
    #
    # elif model_name in "tagcn_ln":
    #     from grb.model.torch.tagcn import TAGCN
    #
    #     model = TAGCN(in_features=num_features,
    #                   out_features=num_classes,
    #                   hidden_features=[128, 128, 128],
    #                   layer_norm=True,
    #                   k=2, activation=F.leaky_relu)
    #     adj_norm_func = utils.normalize.GCNAdjNorm
    #
    # elif model_name in "appnp":
    #     from grb.model.torch.appnp import APPNP
    #
    #     model = APPNP(in_features=num_features,
    #                   out_features=num_classes,
    #                   hidden_features=128,
    #                   alpha=0.01, k=10)
    #     adj_norm_func = utils.normalize.GCNAdjNorm
    #
    # elif model_name in "appnp_ln":
    #     from grb.model.torch.appnp import APPNP
    #
    #     model = APPNP(in_features=num_features,
    #                   out_features=num_classes,
    #                   hidden_features=128,
    #                   layer_norm=True,
    #                   alpha=0.01, k=10)
    #     adj_norm_func = utils.normalize.GCNAdjNorm
    #
    # elif model_name in "gin":
    #     from grb.model.torch.gin import GIN
    #
    #     model = GIN(in_features=num_features,
    #                 out_features=num_classes,
    #                 hidden_features=[128, 128, 128],
    #                 activation=F.relu)
    #     adj_norm_func = utils.normalize.GCNAdjNorm
    #
    # elif model_name in "gin_ln":
    #     from grb.model.torch.gin import GIN
    #
    #     model = GIN(in_features=num_features,
    #                 out_features=num_classes,
    #                 hidden_features=[128, 128, 128],
    #                 layer_norm=True,
    #                 activation=F.relu)
    #     adj_norm_func = utils.normalize.GCNAdjNorm
    #
    # elif model_name in "gat":
    #     from grb.model.dgl.gat import GAT
    #
    #     model = GAT(in_features=num_features,
    #                 out_features=num_classes,
    #                 hidden_features=[64, 64, 64],
    #                 num_heads=4,
    #                 layer_norm=False,
    #                 activation=F.leaky_relu)
    #     adj_norm_func = utils.normalize.GCNAdjNorm
    #
    # elif model_name in "gat_ln":
    #     from grb.model.dgl.gat import GAT
    #
    #     model = GAT(in_features=num_features,
    #                 out_features=num_classes,
    #                 hidden_features=[64, 64, 64],
    #                 num_heads=4,
    #                 layer_norm=True,
    #                 activation=F.leaky_relu)
    #     adj_norm_func = utils.normalize.GCNAdjNorm
    # elif model_name in "gcnguard":
    #     from grb.defense.gnnguard import GCNGuard
    #
    #     model = GCNGuard(in_features=num_features,
    #                      out_features=num_classes,
    #                      hidden_features=[128, 128, 128],
    #                      activation=F.relu,
    #                      drop=True)
    #     adj_norm_func = utils.normalize.GCNAdjNorm
    # elif model_name in "gatguard":
    #     from grb.defense.gnnguard import GATGuard
    #
    #     model = GATGuard(in_features=num_features,
    #                      out_features=num_classes,
    #                      hidden_features=[64, 64, 64],
    #                      num_heads=4,
    #                      activation=F.relu,
    #                      drop=True)
    #     adj_norm_func = utils.normalize.GCNAdjNorm
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
                       inject_mode='tdgia',
                       device=device)
    else:
        raise NotImplementedError

    return attack
