import argparse
import os
import pickle
import sys

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

sys.path.append('../../')

import grb.utils as utils
from grb.dataset.dataset import Dataset, CustomDataset
from grb.utils import normalize, evaluator

import build_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training GNN models')
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--n_epoch", type=int, default=2000, help="Training epoch.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--save_after", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="grb-cora")
    parser.add_argument("--dataset_mode", type=str, default="easy")
    parser.add_argument("--data_dir", type=str, default="/home/stanislas/Research/GRB/data/grb-cora/")
    parser.add_argument("--model_dir", type=str, default="/home/stanislas/Research/GRB/saved_models/grb-cora/")
    parser.add_argument("--model_list", nargs='+', default=["gcn", "gcn_ln", "graphsage", "sgcn",
                                                            "robustgcn", "tagcn", "appnp", "gin"])
    parser.add_argument("--save_name", type=str, default="checkpoint.pt")

    parser.add_argument("--attack_list", nargs='+', default=["fgsm", "pgd", "speit", "tdgia"])
    parser.add_argument("--attack_name", type=str, default="fgsm")
    parser.add_argument("--attack_dir", type=str, default="/home/stanislas/Research/GRB/results/grb-cora/fgsm_vs_gcn")
    parser.add_argument("--save_dir", type=str, default="/home/stanislas/Research/GRB/results/grb-cora")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.gpu >= 0:
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    dataset = Dataset(name=args.dataset,
                      data_dir=args.data_dir,
                      mode=args.dataset_mode,
                      verbose=True)

    adj = dataset.adj
    features = dataset.features
    labels = dataset.labels
    num_nodes = dataset.num_nodes
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask

    if args.attack_dir != '':
        features_attack = np.load(os.path.join(args.attack_dir + "_" + args.dataset_mode, "features.npy"))
        with open(os.path.join(args.attack_dir + "_" + args.dataset_mode, "adj.pkl"), 'rb') as f:
            adj_attack = pickle.load(f)
        adj_attack = sp.csr_matrix(adj_attack)
        adj_attacked = sp.vstack([adj, adj_attack[:, :num_nodes]])
        adj_attacked = sp.hstack([adj_attacked, adj_attack.T])
        adj_attacked = sp.csr_matrix(adj_attacked)
        features_attacked = np.concatenate([features, features_attack])

        dataset = CustomDataset(adj=adj_attacked,
                                features=features_attacked,
                                labels=labels,
                                train_mask=train_mask,
                                val_mask=val_mask,
                                test_mask=test_mask,
                                name=args.dataset)

    else:
        adj_attacked = sp.csr_matrix(adj)
        dataset = CustomDataset(adj=adj_attacked,
                                features=features,
                                labels=labels,
                                train_mask=train_mask,
                                val_mask=val_mask,
                                test_mask=test_mask,
                                name=args.dataset)

    adj = dataset.adj
    features = dataset.features
    labels = dataset.labels
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    features = torch.FloatTensor(features).to(device)
    labels = torch.LongTensor(labels).to(device)

    test_acc_dict = {}
    for model_name in args.model_list:
        model, adj_norm_func = build_model.build_model(model_name=model_name,
                                                       num_features=num_features,
                                                       num_classes=num_classes)
        model.load_state_dict(torch.load(os.path.join(args.model_dir, model_name, "checkpoint.pt")))
        model.to(device)
        model.eval()
        adj_attacked_tensor = utils.adj_preprocess(adj_attacked, adj_norm_func, device)
        logits = model(features, adj_attacked_tensor, dropout=0)
        logp = F.softmax(logits[:num_nodes], 1)
        test_acc = evaluator.eval_acc(logp, labels, test_mask)
        test_acc_dict[model_name] = test_acc.cpu().numpy()
        print("Test score of {}: {:.4f}".format(model_name, test_acc))

    # print("Test ACC dict:", test_acc_dict)
    test_acc_sorted = sorted(list(test_acc_dict.values()))
    final_score = 0.0
    weights = evaluator.get_weights_arithmetic(n=len(args.model_list), w_1=0.005)
    for i in range(len(weights)):
        final_score += weights[i] * test_acc_sorted[i]

    print("Average score: {:.4f}".format(np.average(test_acc_sorted)))
    print("3-max score: {:.4f}".format(np.average(test_acc_sorted[-3:])))
    print("Weighted score: {:.4f}".format(final_score))
