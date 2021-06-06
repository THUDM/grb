import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.append('../')

import grb.utils as utils
from grb.dataset import Dataset
from grb.evaluator import AttackEvaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating adversarial attack against GNNs')
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--dataset", type=str, default="grb-cora")
    parser.add_argument("--dataset_mode", nargs='+', default=["easy", "medium", "hard", "full"])
    parser.add_argument("--feat_norm", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="../data/grb-cora/")
    parser.add_argument("--model", nargs='+', default=None)
    parser.add_argument("--model_dir", type=str, default="../saved_models/grb-cora/")
    parser.add_argument("--model_file", type=str, default="checkpoint.pt")
    parser.add_argument("--config_dir", type=str, default="./grb-cora")
    parser.add_argument("--attack_dir", type=str, default="../results/grb-cora/")
    parser.add_argument("--attack_adj_name", type=str, default="adj.pkl")
    parser.add_argument("--attack_feat_name", type=str, default="features.npy")
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

    result_dict = {"no_attack": {}}
    if args.attack_dir:
        for attack_name in config.attack_list:
            result_dict[attack_name] = {}
    for dataset_mode in args.dataset_mode:
        dataset = Dataset(name=args.dataset,
                          data_dir=args.data_dir,
                          mode=dataset_mode,
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
            model_list = args.model
        else:
            model_list = config.model_list

        model_dict = {}
        for model_name in model_list:
            # Corresponding model path
            model_dict[model_name] = os.path.join(args.model_dir, model_name, args.model_file)

        attack_dict = {}
        for attack_name in config.attack_list:
            for model_sur in config.model_sur_list:
                attack_dict[attack_name] = os.path.join(args.attack_dir,
                                                        attack_name + "_vs_" + model_sur +
                                                        "_" + dataset_mode)
        if args.save_dir is not None:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

        evaluator = AttackEvaluator(dataset=dataset,
                                    build_model=config.build_model,
                                    device=device)

        adj_no = sp.csr_matrix(adj)
        features_no = utils.feat_preprocess(features=features, device=device)

        test_score_dict = evaluator.eval_attack(model_dict=model_dict,
                                                adj_attack=adj_no,
                                                features_attack=features_no)
        result_dict["no_attack"][dataset_mode] = test_score_dict
        if args.attack_dir:
            test_score_dfs_tmp = []
            for attack_name in attack_dict:
                print("Evaluating {} attack..........".format(attack_name))
                features_attack = np.load(os.path.join(attack_dict[attack_name], args.attack_feat_name))
                with open(os.path.join(attack_dict[attack_name], args.attack_adj_name), 'rb') as f:
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
                result_dict[attack_name][dataset_mode] = test_score_dict
    sorted_result_keys = sorted(result_dict, key=lambda x: (result_dict[x]['full']['weighted']))
    result_df = pd.DataFrame.from_dict({(i, j): result_dict[i][j]
                                        for i in sorted_result_keys
                                        for j in result_dict[i].keys()},
                                       orient='index')

    # Calculate model-wise scores, 'average', '3-top', 'weighted'
    eval_dict = {'average': {}, '3-min': {}, 'weighted': {}}
    for i, dataset_mode in enumerate(args.dataset_mode):
        for key in eval_dict.keys():
            eval_dict[key][dataset_mode] = {}
        for model_name in model_list:
            model_score_sorted = sorted(list(result_df[model_name][i::len(args.dataset_mode)].values))
            eval_dict['average'][dataset_mode][model_name] = np.mean(model_score_sorted)
            eval_dict['3-min'][dataset_mode][model_name] = np.mean(model_score_sorted[:3])
            eval_dict['weighted'][dataset_mode][model_name] = evaluator.eval_metric(model_score_sorted,
                                                                                    metric_type='polynomial', order='d')
    sorted_eval_keys = sorted(eval_dict['weighted']['full'],
                              key=lambda x: (eval_dict['weighted']['full'][x]),
                              reverse=True)
    eval_df = pd.DataFrame.from_dict({(i, j): eval_dict[i][j]
                                      for i in eval_dict.keys()
                                      for j in eval_dict[i].keys()},
                                     orient='index')

    result_df = result_df.append(eval_df)
    result_df = result_df[sorted_eval_keys + list(result_df.columns)[-3:]]
    for name in result_df.columns:
        result_df[name] = pd.to_numeric(result_df[name] * 100,
                                        errors='ignore').map('{:,.2f}'.format)
    if args.save_dir is not None:
        result_dict.update(eval_dict)
        utils.save_dict_to_json(result_dict=result_dict,
                                file_dir=args.save_dir,
                                file_name="{}.json".format(args.dataset))
        utils.save_df_to_xlsx(df=result_df,
                              file_dir=args.save_dir,
                              file_name="{}.xlsx".format(args.dataset),
                              verbose=True)
        utils.save_df_to_csv(df=result_df,
                             file_dir=args.save_dir,
                             file_name="{}.csv".format(args.dataset))
        print("Test scores saved in {}.".format(args.save_dir))
    else:
        pd.set_option('display.width', 1000)
        print(result_df)

    print("Evaluation finished.")
