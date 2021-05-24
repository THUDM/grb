import torch.nn.functional as F

from grb.utils import normalize

model_list = ["gcn", "gcn_ln", "graphsage", "sgcn",
              "robustgcn", "tagcn", "appnp", "gin"]

attack_list = ["rnd", "fgsm", "pgd", "speit", "tdgia"]

model_sur_list = ["gcn"]


def build_model(model_name, num_features, num_classes):
    if model_name in "gcn":
        from grb.model.torch.gcn import GCN

        model = GCN(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=[64, 64, 64],
                    activation=F.relu)
        adj_norm_func = normalize.GCNAdjNorm
    elif model_name in "gcn_ln":
        from grb.model.torch.gcn import GCN

        model = GCN(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=[64, 64, 64],
                    layer_norm=True,
                    activation=F.relu)
        adj_norm_func = normalize.GCNAdjNorm
    elif model_name in "graphsage":
        from grb.model.torch.graphsage import GraphSAGE

        model = GraphSAGE(in_features=num_features,
                          out_features=num_classes,
                          hidden_features=[64, 64, 64],
                          activation=F.relu)
        adj_norm_func = normalize.SAGEAdjNorm
    elif model_name in "sgcn":
        from grb.model.torch.sgcn import SGCN

        model = SGCN(in_features=num_features,
                     out_features=num_classes,
                     hidden_features=[64, 64, 64],
                     activation=F.relu)
        adj_norm_func = normalize.GCNAdjNorm
    elif model_name in "robustgcn":
        from grb.model.torch.robustgcn import RobustGCN

        model = RobustGCN(in_features=num_features,
                          out_features=num_classes,
                          hidden_features=[64, 64, 64])
        adj_norm_func = normalize.RobustGCNAdjNorm
    elif model_name in "tagcn":
        from grb.model.torch.tagcn import TAGCN

        model = TAGCN(in_features=num_features,
                      out_features=num_classes,
                      hidden_features=[64, 64, 64],
                      k=2, activation=F.leaky_relu)
        adj_norm_func = normalize.GCNAdjNorm
    elif model_name in "appnp":
        from grb.model.torch.appnp import APPNP

        model = APPNP(in_features=num_features,
                      out_features=num_classes,
                      hidden_features=64,
                      alpha=0.01, k=10)
        adj_norm_func = normalize.GCNAdjNorm
    elif model_name in "gin":
        from grb.model.torch.gin import GIN

        model = GIN(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=[64, 64, 64],
                    activation=F.relu)
        adj_norm_func = normalize.GCNAdjNorm

    return model, adj_norm_func


def build_attack(attack_name, dataset, device="cpu", args=None):
    if attack_name in "rnd":
        from grb.attack.rnd import RND

        config = {}
        config['feat_lim_min'] = args.feat_lim_min
        config['feat_lim_max'] = args.feat_lim_max
        config['n_inject_max'] = args.n_inject
        config['n_edge_max'] = args.n_edge_max

        attack = RND(dataset, adj_norm_func=None, device=device)
        attack.set_config(**config)
    elif attack_name in "fgsm":
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
    elif attack_name in "pgd":
        from grb.attack.pgd import PGD

        config = {}
        config['epsilon'] = args.lr
        config['n_epoch'] = args.n_epoch
        config['feat_lim_min'] = args.feat_lim_min
        config['feat_lim_max'] = args.feat_lim_max
        config['n_inject_max'] = args.n_inject
        config['n_edge_max'] = args.n_edge_max

        attack = PGD(dataset, adj_norm_func=None, device=device)
        attack.set_config(**config)
    elif attack_name in "speit":
        from grb.attack.speit import SPEIT

        config = {}
        config['inject_mode'] = 'random'
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
