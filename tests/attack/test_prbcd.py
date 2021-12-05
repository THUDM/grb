from grb.dataset import Dataset
import grb.utils as utils
import sys

import torch
import torch.nn.functional as F

sys.path.append('..')


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
                hidden_features=128,
                n_layers=3,
                layer_norm=False,
                dropout=0.6)

    print("Number of parameters: {}.".format(utils.get_num_params(model)))
    print(model)

    save_dir = "./saved_models/{}/{}".format('grb-cora', 'gcn')
    save_name = "model_sur.pt"
    device = "cuda:0"
    feat_norm = None
    train_mode = "inductive"  # "transductive"

    from grb.trainer.trainer import Trainer

    trainer = Trainer(dataset=dataset,
                      optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
                      loss=torch.nn.functional.cross_entropy,
                      lr_scheduler=False,
                      early_stop=True,
                      early_stop_patience=500,
                      feat_norm=feat_norm,
                      device=device)
    trainer.train(model=model,
                  n_epoch=5000,
                  eval_every=1,
                  save_after=0,
                  save_dir=save_dir,
                  save_name=save_name,
                  train_mode=train_mode,
                  verbose=False)

    test_score = trainer.evaluate(model, dataset.test_mask)
    print("Test score of surrogate model: {:.4f}".format(test_score))

    # Prepare attack
    from grb.attack.modification.prbcd import PRBCD
    from grb.utils.normalize import GCNAdjNorm

    #n_edge_test = adj[test_mask].getnnz()
    n_mod_ratio = 0.1
    n_node_mod = int(adj.shape[0] * n_mod_ratio)
    n_edge_mod = int(dataset.num_edges * n_mod_ratio)

    attack = PRBCD(epsilon=0.01,
                   n_epoch=500,
                   n_node_mod=n_node_mod,
                   n_edge_mod=0,
                   feat_lim_min=-1,
                   feat_lim_max=1,
                   device='cpu',
                   early_stop=True)

    adj_attack, features_attack = attack.attack(model,
                                                adj,
                                                features,
                                                test_mask,
                                                adj_norm_func=GCNAdjNorm)

    trainer.adj, trainer.features = adj_attack, features_attack
    test_score = trainer.evaluate(model, dataset.test_mask)
    print("Test score of attack on features: {:.4f}".format(test_score))

    attack = PRBCD(epsilon=0.01,
                   n_epoch=500,
                   n_node_mod=0,
                   n_edge_mod=n_edge_mod,
                   feat_lim_min=-1,
                   feat_lim_max=1,
                   device=device,
                   early_stop=True)

    adj_attack, features_attack = attack.attack(model,
                                                adj,
                                                features,
                                                test_mask,
                                                adj_norm_func=GCNAdjNorm)

    trainer.adj, trainer.features = adj_attack, features_attack
    test_score = trainer.evaluate(model, dataset.test_mask)
    print("Test score of attack on edges: {:.4f}".format(test_score))

    attack = PRBCD(epsilon=0.01,
                   n_epoch=500,
                   n_node_mod=n_node_mod,
                   n_edge_mod=n_edge_mod,
                   feat_lim_min=-1,
                   feat_lim_max=1,
                   device=device,
                   early_stop=True)

    adj_attack, features_attack = attack.attack(model,
                                                adj,
                                                features,
                                                test_mask,
                                                adj_norm_func=GCNAdjNorm)

    trainer.adj, trainer.features = adj_attack, features_attack
    test_score = trainer.evaluate(model, dataset.test_mask)
    print("Test score of attack (combined): {:.4f}".format(test_score))
