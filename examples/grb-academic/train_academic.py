import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append('../../')

from grb.dataset.dataset import Dataset
from grb.model.trainer import Trainer

import build_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training GNN models')
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--n_epoch", type=int, default=15000, help="Training epoch.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--save_after", type=int, default=2000)
    parser.add_argument("--dataset", type=str, default="grb-academic")
    parser.add_argument("--data_dir", type=str, default="/home/stanislas/Research/GRB/data/grb-academic/")
    parser.add_argument("--model_dir", type=str, default="/home/stanislas/Research/GRB/saved_models/grb-academic/")
    parser.add_argument("--model_list", nargs='+', default=["gcn", "gcn_ln", "graphsage", "sgcn",
                                                            "robustgcn", "tagcn", "appnp", "gin"])
    parser.add_argument("--save_name", type=str, default="checkpoint.pt")
    parser.add_argument("--early_stop", action='store_true')
    parser.add_argument("--lr_scheduler", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.gpu >= 0:
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    dataset = Dataset(name=args.dataset,
                      data_dir=args.data_dir,
                      mode='normal',
                      verbose=True)

    adj = dataset.adj
    features = dataset.features
    labels = dataset.labels
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    if args.model_list is not None:
        model_list = args.model_list
    else:
        model_list = [args.model_name]
    for model_name in model_list:
        model, adj_norm_func = build_model.build_model(model_name=model_name,
                                                       num_features=num_features,
                                                       num_classes=num_classes)

        adam = torch.optim.Adam(model.parameters(), lr=args.lr)

        trainer = Trainer(dataset=dataset,
                          optimizer=adam,
                          loss=F.nll_loss,
                          adj_norm_func=adj_norm_func,
                          lr_scheduler=args.lr_scheduler,
                          early_stop=args.early_stop,
                          device=device)

        trainer.train(model=model,
                      n_epoch=args.n_epoch,
                      save_dir=os.path.join(args.model_dir, model_name),
                      save_name=args.save_name,
                      eval_every=args.eval_every,
                      save_after=args.save_after,
                      dropout=args.dropout,
                      verbose=args.verbose)

        model.load_state_dict(torch.load(os.path.join(args.model_dir, model_name, args.save_name)))
        _, test_acc = trainer.inference(model)

        print("*" * 80)
        print("Test ACC of {}: {:.4f}".format(model_name, test_acc))
