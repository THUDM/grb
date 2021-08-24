import argparse
import os
import sys
import torch

from grb.dataset import Dataset
from grb.dataset import GRB_SUPPORTED_DATASETS
from grb.utils import Trainer, Logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training GNN models in pipeline.')
    # Dataset settings
    parser.add_argument("--dataset", type=str, default="grb-cora")
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--feat_norm", type=str, default="arctan")
    # Model settings
    parser.add_argument("--model", nargs='+', default=None)
    parser.add_argument("--save_dir", type=str, default="../saved_models/")
    parser.add_argument("--config_dir", type=str, default="../pipeline/configs/")
    parser.add_argument("--log_dir", type=str, default="../pipeline/logs/")
    parser.add_argument("--save_name", type=str, default="model.pt")
    # Training settings
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--n_train", type=int, default=1)
    parser.add_argument("--n_epoch", type=int, default=5000, help="Training epoch.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_after", type=int, default=0)
    parser.add_argument("--train_mode", type=str, default="inductive")
    parser.add_argument("--eval_metric", type=str, default="acc")
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=500)
    parser.add_argument("--lr_scheduler", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.gpu >= 0:
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    if args.dataset not in args.data_dir:
        args.data_dir = os.path.join(args.data_dir, args.dataset)
    if args.dataset not in args.save_dir:
        args.save_dir = os.path.join(args.save_dir, args.dataset)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
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

    if args.model is not None:
        model_list = args.model
    else:
        model_list = config.model_list_basic
    terminal_out = sys.stdout
    for model_name in model_list:
        logger = Logger(file_dir=args.log_dir,
                        file_name="train_{}.out".format(model_name),
                        stream=terminal_out)
        sys.stdout = logger
        print("*" * 80)
        print("Training {} model...........".format(model_name))
        for i in range(args.n_train):
            print("{} time training..........".format(i))
            save_name = args.save_name.split('.')[0] + "_{}.pt".format(i)
            model, train_params = config.build_model(model_name=model_name,
                                                     num_features=dataset.num_features,
                                                     num_classes=dataset.num_classes)

            optimizer = config.build_optimizer(model=model,
                                               lr=train_params["lr"] if "lr" in train_params else args.lr)
            loss = config.build_loss()
            eval_metric = config.build_metric()

            trainer = Trainer(dataset=dataset,
                              optimizer=optimizer,
                              loss=loss,
                              lr_scheduler=args.lr_scheduler,
                              eval_metric=eval_metric,
                              early_stop=train_params[
                                  "early_stop"] if "early_stop" in train_params else args.early_stop,
                              early_stop_patience=train_params[
                                  "early_stop_patience"] if "early_stop_patience" in train_params else args.early_stop_patience,
                              device=device)

            trainer.train(model=model,
                          n_epoch=train_params["n_epoch"] if "n_epoch" in train_params else args.n_epoch,
                          save_dir=os.path.join(args.save_dir, model_name),
                          save_name=save_name,
                          eval_every=args.eval_every,
                          save_after=args.save_after,
                          train_mode=train_params["train_mode"] if "train_mode" in train_params else args.train_mode,
                          verbose=args.verbose)

            model = torch.load(os.path.join(args.save_dir, model_name, save_name), map_location=device)
            val_score = trainer.evaluate(model, dataset.val_mask)
            test_score = trainer.evaluate(model, dataset.test_mask)

            print("*" * 80)
            print("Val ACC of {}: {:.4f}".format(model_name, val_score))
            print("Test ACC of {}: {:.4f}".format(model_name, test_score))

            del model, trainer

    print("Training completed.")
