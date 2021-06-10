import sys

import torch
import torch.nn.functional as F

sys.path.append('../../')

import grb.utils as utils
from grb.dataset import Dataset

dataset = Dataset(name='grb-cora',
                  data_dir="../data/grb-cora",
                  mode='easy', feat_norm="arctan")

adj = dataset.adj
features = dataset.features
labels = dataset.labels
num_nodes = dataset.num_nodes
num_features = dataset.num_features
num_classes = dataset.num_classes
train_mask = dataset.train_mask
test_mask = dataset.test_mask
val_mask = dataset.val_mask

# from grb.model.dgl.gcn import GCN
# from grb.utils.normalize import GCNAdjNorm
#
# model = GCN(in_features=num_features,
#             out_features=num_classes,
#             hidden_features=[64, 64],
#             activation=F.relu)
#
# print("Number of parameters: {}.".format(utils.get_num_params(model)))
# print(model)


# from grb.model.dgl import GAT
# from grb.utils.normalize import GCNAdjNorm
#
# model = GAT(in_features=num_features,
#             out_features=num_classes,
#             hidden_features=[64, 64],
#             num_heads=4,
#             layer_norm=True,
#             activation=F.leaky_relu)

from grb.model.dgl.grand import GRAND
from grb.utils.normalize import GCNAdjNorm

model = GRAND(in_features=num_features,
              out_features=num_classes,
              hidden_features=64)

print("Number of parameters: {}.".format(utils.get_num_params(model)))
print(model)

from grb.utils.trainer import Trainer

adam = torch.optim.Adam(model.parameters(), lr=0.01)
trainer = Trainer(dataset=dataset,
                  optimizer=adam,
                  loss=F.nll_loss,
                  adj_norm_func=GCNAdjNorm,
                  lr_scheduler=False,
                  early_stop=False,
                  device='cuda:0')

trainer.train(model=model,
              n_epoch=1000,
              save_dir="../saved_models/gcn_cora",
              eval_every=5,
              save_after=1,
              dropout=0.5,
              verbose=True)

_, test_acc = trainer.inference(model)
print("Test accuracy: {:.4f}".format(test_acc))

from grb.attack.fgsm import FGSM

fgsm = FGSM(epsilon=0.01,
            n_epoch=10,
            n_inject_max=100,
            n_edge_max=20,
            feat_lim_min=-1,
            feat_lim_max=1,
            device='cpu')
adj_attack, features_attack = fgsm.attack(model=model,
                                          adj=adj,
                                          features=features,
                                          target_mask=test_mask,
                                          adj_norm_func=GCNAdjNorm)
