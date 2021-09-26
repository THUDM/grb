import os
import torch
import argparse
import grb.utils as utils
import torch.nn.functional as F

from grb.dataset import Dataset
from grb.trainer import Trainer


def get_model(model_name):
    if model_name == "grand":
        from grb.model.dgl import GRAND

        model = GRAND(in_features=dataset.num_features,
                      out_features=dataset.num_classes,
                      hidden_features=64,
                      n_layers=2,
                      s=2,
                      k=8,
                      input_dropout=0.5,
                      node_dropout=0.5,
                      hidden_dropout=0.5)

        return model
    if model_name == "gat":
        from grb.model.torch.gat import GAT

        model = GAT(in_features=dataset.num_features,
                    out_features=dataset.num_classes,
                    hidden_features=64,
                    n_layers=3,
                    n_heads=6,
                    activation=F.leaky_relu,
                    layer_norm=False)
        return model
    if model_name == "gat_dgl":
        from grb.model.dgl import GAT

        model = GAT(in_features=dataset.num_features,
                    out_features=dataset.num_classes,
                    hidden_features=64,
                    n_layers=3,
                    n_heads=6,
                    activation=F.leaky_relu,
                    adj_norm_func=None,
                    layer_norm=False)
        return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test GNN model')

    # Dataset settings
    parser.add_argument("--dataset_name", type=str, default="grb-cora")
    parser.add_argument("--data_dir", type=str, default="../../data/")
    parser.add_argument("--feat_norm", type=str, default="arctan")

    # Model settings
    parser.add_argument("--model_name", type=str, default="gat")
    parser.add_argument("--save_dir", type=str, default="./saved_models/")
    parser.add_argument("--save_name", type=str, default="model.pt")

    # Training settings
    parser.add_argument("--n_epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--train_mode", type=str, default="transductive")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.gpu >= 0:
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    # Load dataset
    args.save_dir = os.path.join(args.save_dir, args.dataset_name, args.model_name)
    dataset = Dataset(name=args.dataset_name,
                      data_dir=args.data_dir,
                      mode='full',
                      feat_norm=args.feat_norm)

    utils.fix_seed(seed=args.seed)
    # Load model
    model = get_model(args.model_name)
    print("Number of parameters: {}.".format(utils.get_num_params(model)))
    print(model)

    # Training
    trainer = Trainer(dataset=dataset,
                      optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
                      loss=torch.nn.functional.cross_entropy,
                      lr_scheduler=False,
                      early_stop=True,
                      early_stop_patience=50,
                      feat_norm=None,
                      device=device)

    trainer.train(model=model,
                  n_epoch=args.n_epoch,
                  eval_every=1,
                  save_after=0,
                  save_dir=args.save_dir,
                  save_name=args.save_name,
                  train_mode=args.train_mode,
                  verbose=False)

    # Inference
    model = torch.load(os.path.join(args.save_dir, args.save_name))
    model = model.to(device)
    model.eval()

    pred = trainer.inference(model)
    test_score = trainer.evaluate(model, dataset.test_mask)
    print("Test score: {:.4f}".format(test_score))
