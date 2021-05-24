import torch
import torch.nn.functional as F

from grb.dataset.dataset import Dataset
from grb.model.torch.gcn import GCN
from grb.evaluator import metric
from grb.attack.tdgia import TDGIA
from grb.utils.normalize import GCNAdjNorm

if __name__ == '__main__':
    device = "cuda:3"
    # Data preparing
    dataset = Dataset('cora')
    adj = dataset.adj
    adj_tensor = dataset.adj_tensor
    features = dataset.features
    labels = dataset.labels
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    # Model loading
    model_path = '../saved_models/gcn_cora/checkpoint.pt'
    model = GCN(in_features=num_features,
                out_features=num_classes,
                hidden_features=[64, 64],
                activation=F.relu)
    model.load_state_dict(torch.load(model_path))
    print("Model loaded.")

    # Prediction test
    pred = model.forward(features, adj_tensor, dropout=0)
    acc = metric.eval_acc(pred, labels, mask=dataset.test_mask)
    print("Test accuracy: {:.4f}".format(acc))

    config = {}
    config['inject_mode'] = 'tdgia'
    config['lr'] = 0.01
    config['n_epoch'] = 100
    config['feat_lim_min'] = 0
    config['feat_lim_max'] = 1
    config['n_inject_max'] = 500
    config['n_edge_max'] = 20

    tdgia = TDGIA(dataset, adj_norm_func=GCNAdjNorm, device=device)
    tdgia.set_config(**config)
    adj_attack, features_attack = tdgia.attack(model)
