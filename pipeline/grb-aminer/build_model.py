import torch.nn.functional as F
from grb.utils import normalize


def build_model(model_name, num_features, num_classes):
    if model_name in "gcn":
        from grb.model.gcn import GCN

        model = GCN(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=[256, 128, 64],
                    activation=F.relu)
        adj_norm_func = normalize.GCNAdjNorm
    elif model_name in "gcn_ln":
        from grb.model.gcn import GCN

        model = GCN(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=[256, 128, 64],
                    layer_norm=True,
                    activation=F.relu)
        adj_norm_func = normalize.GCNAdjNorm
    elif model_name in "graphsage":
        from grb.model.graphsage import GraphSAGE

        model = GraphSAGE(in_features=num_features,
                          out_features=num_classes,
                          hidden_features=[128, 128, 128],
                          activation=F.relu)
        adj_norm_func = normalize.SAGEAdjNorm
    elif model_name in "sgcn":
        from grb.model.sgcn import SGCN

        model = SGCN(in_features=num_features,
                     out_features=num_classes,
                     hidden_features=[128, 128, 128, 128],
                     activation=F.relu)
        adj_norm_func = normalize.GCNAdjNorm
    elif model_name in "robustgcn":
        from grb.model.robustgcn import RobustGCN

        model = RobustGCN(in_features=num_features,
                          out_features=num_classes,
                          hidden_features=[128, 128, 128])
        adj_norm_func = normalize.RobustGCNAdjNorm
    elif model_name in "tagcn":
        from grb.model.tagcn import TAGCN

        model = TAGCN(in_features=num_features,
                      out_features=num_classes,
                      hidden_features=[128, 128, 128],
                      k=3, activation=F.leaky_relu)
        adj_norm_func = normalize.GCNAdjNorm
    elif model_name in "appnp":
        from grb.model.appnp import APPNP

        model = APPNP(in_features=num_features,
                      out_features=num_classes,
                      hidden_features=128,
                      alpha=0.01, k=10)
        adj_norm_func = normalize.GCNAdjNorm
    elif model_name in "gin":
        from grb.model.gin import GIN

        model = GIN(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=[128, 128, 128, 128],
                    activation=F.relu)
        adj_norm_func = normalize.GCNAdjNorm

    return model, adj_norm_func
