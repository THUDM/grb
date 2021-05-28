import torch
import torch.nn.functional as F

import sys

sys.path.append('..')

import grb.utils as utils
from grb.dataset import Dataset

if __name__ == '__main__':
    # Load data
    dataset = Dataset(name='grb-cora',
                      data_dir="../data/grb-cora",
                      mode='easy', feat_norm="arctan")

    adj = dataset.adj
    features = dataset.features
    labels = dataset.labels
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    # Load model
    from grb.model.torch.gcn import GCN

    model = GCN(in_features=num_features,
                out_features=num_classes,
                hidden_features=[64, 64],
                activation=F.relu)

    print("Number of parameters: {}.".format(utils.get_num_params(model)))
    print(model)

    # Define trainer
    from grb.utils.trainer import Trainer
    from grb.utils.normalize import GCNAdjNorm

    device = 'cuda:0'
    adam = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(dataset=dataset,
                      optimizer=adam,
                      loss=F.nll_loss,
                      adj_norm_func=GCNAdjNorm,
                      lr_scheduler=False,
                      early_stop=False,
                      device=device)

    # Training
    trainer.train(model=model,
                  n_epoch=10,
                  save_dir="../saved_models/gcn_cora",
                  eval_every=1,
                  save_after=0,
                  train_mode="inductive",
                  dropout=0.5,
                  verbose=True)

    _, test_acc = trainer.inference(model)
    print("Test accuracy: {:.4f}".format(test_acc))

    print("Model test passed.")
