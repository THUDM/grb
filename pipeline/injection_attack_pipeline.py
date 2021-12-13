import argparse
import os
import sys
import time

import torch

import grb.utils as utils
from grb.dataset import Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Injection attack on GML models.')

    # Dataset settings
    parser.add_argument("--dataset", type=str, default="grb-cora")
    parser.add_argument("--dataset_mode", nargs='+', default=["easy", "medium", "hard", "full"])
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--feat_norm", type=str, default="arctan")

    # Model settings
    parser.add_argument("--model", nargs='+', default=None)
    parser.add_argument("--model_dir", type=str, default="../saved_models/")
    parser.add_argument("--config_dir", type=str, default="../pipeline/configs/")
    parser.add_argument("--log_dir", type=str, default="../pipeline/logs/")
    parser.add_argument("--model_file", type=str, default="model_sur.pt")

    # Injection attack setting
    parser.add_argument("--attack", nargs='+', default=None)
    parser.add_argument("--attack_mode", type=str, default="injection")
    parser.add_argument("--save_dir", type=str, default="../attack_results/")
    parser.add_argument("--attack_epoch", type=int, default=500)
    parser.add_argument("--attack_lr", type=float, default=0.01)
    parser.add_argument("--n_attack", type=int, default=1)
    parser.add_argument("--n_inject_ratio", type=float, default=None)
    parser.add_argument("--n_inject_max", type=int, default=20)
    parser.add_argument("--n_edge_max", type=int, default=20)
    parser.add_argument("--feat_lim_min", type=float, default=None)
    parser.add_argument("--feat_lim_max", type=float, default=None)
    parser.add_argument("--flip_type", type=str, default="deg")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.gpu >= 0:
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    if args.dataset not in args.data_dir:
        args.data_dir = os.path.join(args.data_dir, args.dataset)
    if args.dataset not in args.model_dir:
        args.model_dir = os.path.join(args.model_dir, args.dataset)
    if args.dataset not in args.save_dir:
        args.save_dir = os.path.join(args.save_dir, args.dataset)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.dataset not in args.config_dir:
        args.config_dir = os.path.join(args.config_dir, args.dataset)
    if args.dataset not in args.log_dir:
        args.log_dir = os.path.join(args.log_dir, args.dataset)

    sys.path.append(args.config_dir)
    import config

    for dataset_mode in args.dataset_mode:
        dataset = Dataset(name=args.dataset,
                          data_dir=args.data_dir,
                          mode=dataset_mode,
                          feat_norm=args.feat_norm,
                          verbose=True)

        adj = dataset.adj
        features = dataset.features
        labels = dataset.labels
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        target_mask = dataset.test_mask

        if args.n_inject_ratio is not None:
            n_test = len(dataset.index_test)
            args.n_inject_max = int(n_test * args.n_inject_ratio)
            args.n_edge_max = int(dataset.num_edges * args.n_inject_ratio / args.n_inject_max)
        if args.feat_lim_min is None:
            args.feat_lim_min = features.min()
        if args.feat_lim_max is None:
            args.feat_lim_max = features.max()

        print("Attack vs. Defense..........")
        print("    Number of injection: {}".format(args.n_inject_max))
        print("    Number of maximum edges for each injected nodes: {}".format(args.n_edge_max))
        print("    Injected feature range: [{:.4f}, {:.4f}]".format(args.feat_lim_min, args.feat_lim_max))

        if args.attack is not None:
            attack_list = args.attack
        else:
            attack_list = config.injection_attack_list
        if args.model is not None:
            model_list = args.model
        else:
            model_list = config.model_sur_list

        for attack_name in attack_list:
            attack = config.build_attack(attack_name=attack_name,
                                         device=device,
                                         args=args,
                                         mode=args.attack_mode)

            for model_name in model_list:
                print("{} vs. {}..........".format(attack_name, model_name))
                model_sur = torch.load(os.path.join(args.model_dir, model_name, args.model_file),
                                       map_location=device)
                print("Model loaded from {}".format(os.path.join(args.model_dir, model_name, args.model_file)))

                for i in range(args.n_attack):
                    time_start = time.time()
                    print("{} attack..........".format(i + 1))
                    adj_attack, features_attack = attack.attack(model=model_sur,
                                                                adj=adj,
                                                                features=features,
                                                                target_mask=target_mask,
                                                                adj_norm_func=model_sur.adj_norm_func)
                    time_end = time.time()
                    print("Attack runtime: {:.4f}".format(time_end - time_start))
                    save_dir = os.path.join(args.save_dir, attack_name + "_vs_" + model_name, dataset_mode, str(i))
                    utils.save_adj(adj_attack.tocsr()[-args.n_inject_max:, :], save_dir)
                    utils.save_features(features_attack, save_dir)

        print("Attack finished.")
