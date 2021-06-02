import sys

import torch.nn.functional as F

sys.path.append('..')

import grb.utils as utils
from grb.dataset import Dataset

if __name__ == '__main__':
    # Load data
    dataset = Dataset(name='grb-cora',
                      data_dir="../../data/grb-cora",
                      mode='easy', feat_norm="arctan")

    adj = dataset.adj
    features = dataset.features
    labels = dataset.labels
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    test_mask = dataset.test_mask

    # Load model
    from grb.model.torch.gcn import GCN

    model = GCN(in_features=num_features,
                out_features=num_classes,
                hidden_features=[64, 64],
                activation=F.relu)

    print("Number of parameters: {}.".format(utils.get_num_params(model)))
    print(model)

    # Prepare attack
    from grb.attack.tdgia import TDGIA
    from grb.utils.normalize import GCNAdjNorm

    device = 'cuda:0'

    attack = TDGIA(lr=0.001,
                   n_epoch=10,
                   n_inject_max=20,
                   n_edge_max=20,
                   feat_lim_min=-0.99,
                   feat_lim_max=0.99,
                   device=device,
                   inject_mode='tdgia')

    adj_attack, features_attack = attack.attack(model=model,
                                                adj=adj,
                                                features=features,
                                                target_mask=test_mask,
                                                adj_norm_func=GCNAdjNorm)
    print(adj_attack, features_attack)
