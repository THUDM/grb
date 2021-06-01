import argparse
import os
import sys

import torch

sys.path.append('../')

from grb.dataset import Dataset
from grb.utils import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training GNN models in pipeline.')
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--n_epoch", type=int, default=2000, help="Training epoch.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--save_after", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="grb-cora")
    parser.add_argument("--feat_norm", type=str, default=None)
    parser.add_argument("--train_mode", type=str, default="inductive")
    parser.add_argument("--data_dir", type=str, default="../data/grb-cora/")
    parser.add_argument("--model_dir", type=str, default="../saved_models/grb-cora/")
    parser.add_argument("--config_dir", type=str, default="./grb-cora")
    parser.add_argument("--model", nargs='+', default=None)
    parser.add_argument("--save_name", type=str, default="checkpoint.pt")
    parser.add_argument("--early_stop", action='store_true')
    parser.add_argument("--lr_scheduler", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--n_train", type=int, default=1)

    args = parser.parse_args()

    if args.gpu >= 0:
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    sys.path.append(args.config_dir)
    import config

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    dataset = Dataset(name=args.dataset,
                      data_dir=args.data_dir,
                      mode='full',
                      feat_norm=args.feat_norm,
                      verbose=True)

    adj = dataset.adj
    features = dataset.features
    labels = dataset.labels
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    if args.model is not None:
        model_list = args.model
    else:
        model_list = config.model_list
    for model_name in model_list:
        print("*" * 80)
        print("Training {} model...........".format(model_name))
        for i in range(args.n_train):
            print("{} time training..........".format(i))
            model, adj_norm_func = config.build_model(model_name=model_name,
                                                      num_features=num_features,
                                                      num_classes=num_classes)

            optimizer = config.build_optimizer(model=model, lr=args.lr)
            loss = config.build_loss()
            eval_metric = config.build_metric()

            trainer = Trainer(dataset=dataset,
                              optimizer=optimizer,
                              loss=loss,
                              adj_norm_func=adj_norm_func,
                              lr_scheduler=args.lr_scheduler,
                              eval_metric=eval_metric,
                              early_stop=args.early_stop,
                              device=device)

            trainer.train(model=model,
                          n_epoch=args.n_epoch,
                          save_dir=os.path.join(args.model_dir, model_name, str(i)),
                          save_name=args.save_name,
                          eval_every=args.eval_every,
                          save_after=args.save_after,
                          train_mode=args.train_mode,
                          dropout=args.dropout,
                          verbose=args.verbose)

            model.load_state_dict(torch.load(os.path.join(args.model_dir, model_name, str(i), args.save_name),
                                             map_location=device))
            _, test_score = trainer.inference(model)

            print("*" * 80)
            print("Test ACC of {}: {:.4f}".format(model_name, test_score))

            del model, trainer

    print("Training completed.")
