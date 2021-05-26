import argparse
import os
import pickle
import sys

import numpy as np
import scipy.sparse as sp

sys.path.append('../')

import grb.utils as utils
from grb.dataset import Dataset
from grb.evaluator import AttackEvaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating adversarial attack against GNNs')
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--dataset", type=str, default="grb-cora")
    parser.add_argument("--dataset_mode", type=str, default="easy")
    parser.add_argument("--feat_norm", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="../data/grb-cora/")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default="../saved_models/grb-cora/")
    parser.add_argument("--model_file", type=str, default="checkpoint.pt")
    parser.add_argument("--config_dir", type=str, default="./grb-cora")
    parser.add_argument("--attack_dir", type=str, default="../results/grb-cora/")
    parser.add_argument("--weight_type", type=str, default="polynomial",
                        help="Type of weighted accuracy, 'polynomial' or 'arithmetic'.")
    parser.add_argument("--save_dir", type=str, default=None)

    args = parser.parse_args()

    if args.gpu >= 0:
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    sys.path.append(args.config_dir)
    import config

    dataset = Dataset(name=args.dataset,
                      data_dir=args.data_dir,
                      mode=args.dataset_mode,
                      feat_norm=args.feat_norm,
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

    if args.model is not None:
        model_list = [args.model]
    else:
        model_list = config.model_list

    model_dict = {}
    for model_name in model_list:
        # Corresponding model path
        model_dict[model_name] = os.path.join(args.model_dir, model_name, args.model_file)

    attack_dict = {}
    for attack_name in config.attack_list:
        for model_sur in config.model_sur_list:
            attack_dict[attack_name + "_" + model_sur] = os.path.join(args.attack_dir,
                                                                      attack_name + "_vs_" + model_sur +
                                                                      "_" + args.dataset_mode)

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    evaluator = AttackEvaluator(dataset=dataset,
                                build_model=config.build_model,
                                device=device)

    if args.attack_dir == '':
        adj = sp.csr_matrix(adj)
        features = utils.feat_preprocess(features=features, device=device)
        labels = utils.label_preprocess(labels=labels, device=device)

        test_score_dict = evaluator.eval_attack(model_dict=model_dict,
                                                adj_attack=adj,
                                                features_attack=features)

        for key, value in test_score_dict.items():
            print("{} score: {:.4f}".format(key, value))

        if args.save_dir is not None:
            file_name = "noattack_{}_{}.xlsx".format(args.dataset, args.dataset_mode)
            utils.save_dict_to_xlsx(result_dict=test_score_dict,
                                    file_dir=args.save_dir,
                                    file_name=file_name,
                                    index=["noattack"])
            print("Test scores saved in {}.".format(os.path.join(args.save_dir, file_name)))
    else:
        for attack_name in attack_dict:
            print("Evaluating {} attack..........".format(attack_name))
            features_attack = np.load(os.path.join(attack_dict[attack_name], "features.npy"))
            with open(os.path.join(attack_dict[attack_name], "adj.pkl"), 'rb') as f:
                adj_attack = pickle.load(f)
            adj_attack = sp.csr_matrix(adj_attack)
            adj_attacked = sp.vstack([adj, adj_attack[:, :num_nodes]])
            adj_attacked = sp.hstack([adj_attacked, adj_attack.T])
            adj_attacked = sp.csr_matrix(adj_attacked)
            features_attacked = np.concatenate([features, features_attack])
            features_attacked = utils.feat_preprocess(features=features_attacked, device=device)
            test_score_dict = evaluator.eval_attack(model_dict=model_dict,
                                                    adj_attack=adj_attacked,
                                                    features_attack=features_attacked)

            for key, value in test_score_dict.items():
                print("{} score: {:.4f}".format(key, value))

            if args.save_dir is not None:
                file_name = "{}_{}_{}.xlsx".format(attack_name, args.dataset, args.dataset_mode)
                utils.save_dict_to_xlsx(result_dict=test_score_dict,
                                        file_dir=args.save_dir,
                                        file_name=file_name,
                                        index=[attack_name])
                print("Test scores saved in {}.".format(os.path.join(args.save_dir, file_name)))

    print("Evaluation finished.")
