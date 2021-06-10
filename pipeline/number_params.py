import argparse
import os
import sys

import torch

sys.path.append('../')

import grb.utils as utils
from grb.dataset import Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attacking GNN models in pipeline.')
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--n_epoch", type=int, default=200, help="Training epoch.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--dataset", type=str, default="grb-cora")
    parser.add_argument("--dataset_mode", nargs='+', default=["easy", "medium", "hard", "full"])
    parser.add_argument("--data_dir", type=str, default="../data/grb-cora")
    parser.add_argument("--config_dir", type=str, default="./grb-cora")
    parser.add_argument("--feat_norm", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default="../saved_models/grb-cora")
    parser.add_argument("--model_file", type=str, default="checkpoint.pt")
    parser.add_argument("--attack", nargs='+', default=None)
    parser.add_argument("--save_dir", type=str, default="../results/test/")
    parser.add_argument("--n_attack", type=int, default=1)
    parser.add_argument("--n_inject", type=int, default=20)
    parser.add_argument("--n_edge_max", type=int, default=20)
    parser.add_argument("--feat_lim_min", type=float)
    parser.add_argument("--feat_lim_max", type=float)
    parser.add_argument("--early_stop", action='store_true')

    args = parser.parse_args()

    if args.gpu >= 0:
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    sys.path.append(args.config_dir)
    import config

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = Dataset(name=args.dataset,
                      data_dir=args.data_dir,
                      feat_norm=args.feat_norm,
                      verbose=True)

    adj = dataset.adj
    features = dataset.features
    labels = dataset.labels
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    target_mask = dataset.test_mask

    if args.model is not None:
        model_list = [args.model]
    else:
        model_list = config.model_list

    for model_name in model_list:
        model, adj_norm_func = config.build_model(model_name=model_name,
                                                  num_features=num_features,
                                                  num_classes=num_classes)
        print(model)
        print(utils.get_num_params(model))
