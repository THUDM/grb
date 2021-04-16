import cogdl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from grb.dataset.dataloader import DataLoader
from grb.model.gcn import GCN
from grb.utils import evaluator
from grb.attack.tdgia import TDGIA

if __name__ == '__main__':
    # Data preparing
    dataset = DataLoader('cora')
    adj = dataset.adj
    adj_tensor = dataset.adj_tensor
    features = dataset.features
    labels = dataset.labels
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    # Model loading
    num_hidden = 64
    model_path = "../grb/model/saved_models/model_gcn_cora.pt"
    model = GCN(num_layers=3, num_features=[num_features, num_hidden, num_hidden, num_classes], activation=F.relu)
    model.load_state_dict(torch.load(model_path))
    print("Model loaded.")

    # Prediction test
    pred = model.forward(features, adj_tensor, dropout=0)
    acc = evaluator.eval_acc(pred, labels, mask=dataset.test_mask)
    print("Test accuracy: {:.4f}".format(acc))

    # Attack configuration
    config = {}
    config['mode'] = 'random-inter'
    config['lr'] = 0.01
    config['n_epoch'] = 10
    config['feat_lim_min'] = 0
    config['feat_lim_max'] = 1

    # Attack test
    tdgia = TDGIA(dataset, n_inject_max=100, n_edge_max=100)
    tdgia.set_config(**config)

    adj_attack, features_attack = tdgia.attack(model, features, adj)
