import os
import argparse
import torch
import grb.utils as utils

from grb.dataset import CogDLDataset
from grb.trainer import GraphTrainer


def get_model(model_name):
    if model_name == "gcngc":
        from grb.model.torch import GCNGC

        model = GCNGC(in_features=dataset.num_features,
                      out_features=dataset.num_classes,
                      hidden_features=64,
                      n_layers=3)
        return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training GNNs for Graph Classification')

    # Dataset settings
    parser.add_argument("--dataset_name", type=str, default="mutag")
    parser.add_argument("--data_dir", type=str, default="../../data/")
    # Model settings
    parser.add_argument("--model_name", type=str, default="gcngc")
    parser.add_argument("--save_dir", type=str, default="./saved_models/")
    parser.add_argument("--save_name", type=str, default="model.pt")

    # Training settings
    parser.add_argument("--n_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.gpu >= 0:
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    # Load dataset
    args.save_dir = os.path.join(args.save_dir, args.dataset_name, args.model_name)
    dataset = CogDLDataset(name=args.dataset_name,
                           data_dir=args.data_dir)

    utils.fix_seed(seed=args.seed)

    # Load model
    model = get_model(args.model_name)
    print("Number of parameters: {}.".format(utils.get_num_params(model)))
    print(model)

    # Training
    trainer = GraphTrainer(dataset=dataset,
                           batch_size=args.batch_size,
                           optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
                           loss=torch.nn.functional.cross_entropy,
                           lr_scheduler=False,
                           early_stop=True,
                           early_stop_patience=50,
                           device=device)

    trainer.train(model=model,
                  n_epoch=200,
                  eval_every=1,
                  save_after=0,
                  save_dir=args.save_dir,
                  save_name=args.save_name,
                  verbose=False)

    # Inference
    model = torch.load(os.path.join(args.save_dir, args.save_name))
    model = model.to(device)
    model.eval()

    pred = trainer.inference(model)
    test_score = trainer.evaluate(model, dataset.index_test)
    print("Test score: {:.4f}".format(test_score))
