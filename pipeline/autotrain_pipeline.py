import argparse
import os
import sys

import grb.utils as utils
from grb.dataset import Dataset
from grb.dataset import GRB_SUPPORTED_DATASETS
from grb.evaluator import metric
from grb.utils import AutoTrainer, Logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto training GNN models in pipeline.')
    # Dataset settings
    parser.add_argument("--dataset", type=str, default="grb-cora")
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--feat_norm", type=str, default="arctan")
    # Model settings
    parser.add_argument("--model", nargs='+', default=None)
    parser.add_argument("--config_dir", type=str, default="../pipeline/configs/")
    parser.add_argument("--log_dir", type=str, default="../pipeline/logs/")
    parser.add_argument("--save_name", type=str, default="model.pt")
    # Training settings
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--n_epoch", type=int, default=5000, help="Training epoch.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_after", type=int, default=0)
    parser.add_argument("--train_mode", type=str, default="inductive")
    parser.add_argument("--eval_metric", type=str, default="acc")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--if_save", action='store_true')

    args = parser.parse_args()

    if args.gpu >= 0:
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"

    if args.dataset not in args.data_dir:
        args.data_dir = os.path.join(args.data_dir, args.dataset)
    if args.dataset not in args.config_dir:
        args.config_dir = os.path.join(args.config_dir, args.dataset)
    if args.dataset not in args.log_dir:
        args.log_dir = os.path.join(args.log_dir, args.dataset)

    print(args)
    sys.path.append(args.config_dir)
    import config

    if args.dataset in GRB_SUPPORTED_DATASETS:
        dataset = Dataset(name=args.dataset,
                          data_dir=args.data_dir,
                          mode='full',
                          feat_norm=args.feat_norm,
                          verbose=True)
    else:
        print("{} dataset not supported.".format(args.dataset))
        exit(1)

    adj = dataset.adj
    features = dataset.features
    labels = dataset.labels
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    if args.eval_metric == "acc":
        args.eval_metric = metric.eval_acc
    elif args.eval_metric == "rocauc":
        args.eval_metric = metric.eval_rocauc
    else:
        args.eval_metric = metric.eval_acc

    if args.model is not None:
        model_list = args.model
    else:
        model_list = config.model_list_basic

    utils.fix_seed(args.seed)
    other_params = {"train_mode": args.train_mode,
                    "eval_every": args.eval_every,
                    "save_after": args.save_after}
    terminal_out = sys.stdout
    for model_name in model_list:
        logger = Logger(file_dir=args.log_dir,
                        file_name="{}.out".format(model_name),
                        stream=terminal_out)
        sys.stdout = logger
        print("*" * 80)
        print("Auto training {} model...........".format(model_name))
        model_class, params_search = config.build_model_autotrain(model_name=model_name)
        auto_trainer = AutoTrainer(dataset=dataset,
                                   model_class=model_class,
                                   eval_metric=args.eval_metric,
                                   params_search=params_search,
                                   n_trials=args.n_trials,
                                   n_jobs=args.n_jobs,
                                   if_save=args.if_save,
                                   seed=args.seed,
                                   device=args.device,
                                   **other_params)

        best_score, best_params, best_score_list = auto_trainer.run()
        print("Best validation score: {:.4f}".format(best_score))
        print("Best parameters: ", best_params)

        del auto_trainer
        logger.flush()

    print("Auto training completed.")
