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
    parser.add_argument("--dataset_mode", type=str, default="easy")
    parser.add_argument("--data_dir", type=str, default="../data/grb-cora")
    parser.add_argument("--config_dir", type=str, default="./grb-cora")
    parser.add_argument("--feat_norm", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default="../saved_models/grb-cora")
    parser.add_argument("--model_file", type=str, default="checkpoint.pt")
    parser.add_argument("--attack", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="../results/test/")
    parser.add_argument("--n_attack", type=int, default=1)
    parser.add_argument("--n_inject", type=int, default=20)
    parser.add_argument("--n_edge_max", type=int, default=20)
    parser.add_argument("--feat_lim_min", type=float, default=-1)
    parser.add_argument("--feat_lim_max", type=float, default=1)

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
                      mode=args.dataset_mode,
                      feat_norm=args.feat_norm,
                      verbose=True)

    adj = dataset.adj
    features = dataset.features
    labels = dataset.labels
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    target_mask = dataset.test_mask

    print("Attack vs. Defense..........")
    if args.attack is not None:
        attack_list = [args.attack]
    else:
        attack_list = config.attack_list
    if args.model is not None:
        model_list = [args.model]
    else:
        model_list = config.model_list

    for attack_name in attack_list:
        attack = config.build_attack(attack_name,
                                     dataset=dataset,
                                     device=device,
                                     args=args)

        for model_name in model_list:
            print("{} vs. {}..........".format(attack_name, model_name))
            model, adj_norm_func = config.build_model(model_name=model_name,
                                                      num_features=num_features,
                                                      num_classes=num_classes)
            model.load_state_dict(
                torch.load(os.path.join(args.model_dir, model_name, args.model_file)))
            print("Model loaded from {}".format(os.path.join(args.model_dir, model_name, args.model_file)))

            for i in range(args.n_attack):
                print("{} attack..........".format(i + 1))
                adj_attack, features_attack = attack.attack(model=model,
                                                            adj=adj,
                                                            features=features,
                                                            target_mask=target_mask,
                                                            adj_norm_func=adj_norm_func)
                save_dir = os.path.join(args.save_dir,
                                        attack_name + "_vs_" + model_name + "_" + args.dataset_mode, str(i))

                utils.save_adj(adj_attack.tocsr()[-args.n_inject:, :], save_dir)
                utils.save_features(features_attack, save_dir)

                # if utils.check_symmetry(adj_attack.tocsr()):
                #     utils.save_adj(adj_attack.tocsr()[-args.n_inject:, :], save_dir)
                # else:
                #     print("The generated adjacency matrix is not symmetric!")
                # if utils.check_feat_range(features_attack,
                #                           feat_lim_min=args.feat_lim_min,
                #                           feat_lim_max=args.feat_lim_max):
                #     utils.save_features(features_attack, save_dir)
                # else:
                #     print("The generated features exceed the feature limit!")

    print("Attack finished.")
