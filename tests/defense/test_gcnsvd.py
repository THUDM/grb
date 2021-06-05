import sys

import torch.nn.functional as F

sys.path.append('..')

import grb.utils as utils
from grb.dataset import Dataset
from grb.defense.gcnsvd import GCNSVD

if __name__ == '__main__':
    device = 'cuda:0'
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

    from grb.utils import fix_seed
    from grb.utils.normalize import GCNAdjNorm

    fix_seed(42)
    model = GCNSVD(in_features=num_features,
                   out_features=num_classes,
                   hidden_features=[64, 64],
                   activation=F.relu)
    model.to(device)
    features = utils.feat_preprocess(features=features,
                                     device=device)
    adj = utils.adj_preprocess(adj=adj,
                               adj_norm_func=GCNAdjNorm,
                               model_type=model.model_type,
                               device=device)
    logits = model(features, adj)

    print("Number of parameters: {}.".format(utils.get_num_params(model)))
