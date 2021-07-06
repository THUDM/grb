import os
import time

import torch
import torch.nn.functional as F

from ..utils import utils
from ..evaluator import metric


class Trainer(object):
    r"""

    Description
    -----------
    Trainer for GNNs.

    Parameters
    ----------
    dataset : grb.dataset.Dataset or grb.dataset.CustomDataset
        GRB supported dataset.
    optimizer : torch.optim
        Optimizer for training.
    loss : func of torch.nn.functional
        Loss function.
    adj_norm_func : func of utils.normalize, optional
        Function that normalizes adjacency matrix. Default: ``None``.
    feat_norm : str, optional
        Type of feature normalization, ['arctan', 'tanh']. Default: ``None``.
    lr_scheduler : bool, optional
        Whether to use learning rate scheduler.
    early_stop : bool, optional
        Whether to use early stop.
    eval_metric : func of grb.metric, optional
        Evaluation metric, like accuracy or F1 score. Default: ``grb.metric.eval_acc``.
    device : str, optional
        Device used to host data. Default: ``cpu``.

    """
    def __init__(self,
                 dataset,
                 optimizer,
                 loss,
                 adj_norm_func=None,
                 feat_norm=None,
                 lr_scheduler=False,
                 early_stop=False,
                 eval_metric=metric.eval_acc,
                 device='cpu'):
        # Load dataset
        self.adj = dataset.adj
        self.raw_features = dataset.features
        self.labels = dataset.labels
        self.train_mask = dataset.train_mask
        self.val_mask = dataset.val_mask
        self.test_mask = dataset.test_mask
        self.num_classes = dataset.num_classes
        self.adj_norm_func = adj_norm_func

        self.device = device
        self.features = utils.feat_preprocess(features=self.raw_features,
                                              feat_norm=feat_norm,
                                              device=self.device)
        self.labels = utils.label_preprocess(labels=self.labels,
                                             device=self.device)

        # Settings
        self.optimizer = optimizer
        self.loss = loss
        self.eval_metric = eval_metric

        # Learning rate scheduling
        if lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=100,
                factor=0.75,
                min_lr=0.0,
                verbose=True)
        else:
            self.lr_scheduler = lr_scheduler

        # Early stop
        if early_stop:
            self.early_stop = EarlyStop()
        else:
            self.early_stop = early_stop

    def train(self,
              model,
              n_epoch,
              save_dir=None,
              save_name=None,
              eval_every=10,
              save_after=0,
              train_mode="trasductive",
              dropout=0.0,
              verbose=True):
        r"""

        Description
        -----------
        Train a GNN model.

        Parameters
        ----------
        model : torch.nn.module
            Model implemented based on ``torch.nn.module``.
        n_epoch : int
            Number of epoch.
        save_dir : str, optional
            Directory for saving model. Default: ``None``.
        save_name : str, optional
            Name for saved model. Default: ``None``.
        eval_every : int, optional
            Evaluation step. Default: ``10``.
        save_after : int, optional
            Save after certain number of epoch. Default: ``0``.
        train_mode : str, optional
            Training mode, ['inductive', 'transductive']. Default: ``transductive``.
        dropout : float, optional
            Rate of dropout. Default: ``0.0``.
        verbose : bool, optional
            Whether to display logs. Default: ``False``.

        """
        model.to(self.device)
        model.train()

        if save_dir is None:
            cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            save_dir = "./tmp_{}".format(cur_time)
        else:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        if save_name is None:
            save_name = "checkpoint.pt"
        else:
            if save_name.split(".")[-1] != "pt":
                save_name = save_name + ".pt"

        train_score_list = []
        val_score_list = []
        best_val_score = 0.0
        features = self.features
        train_mask = self.train_mask
        val_mask = self.val_mask
        labels = self.labels

        if train_mode == "inductive":
            # Inductive setting
            train_val_mask = torch.logical_or(train_mask, val_mask)
            train_val_index = torch.where(train_val_mask)[0]
            train_index, val_index = torch.where(train_mask)[0], torch.where(val_mask)[0]
            train_index_induc, val_index_induc = utils.get_index_induc(train_index, val_index)
            train_mask_induc = torch.zeros(len(train_val_index), dtype=bool)
            train_mask_induc[train_index_induc] = True
            val_mask_induc = torch.zeros(len(train_val_index), dtype=bool)
            val_mask_induc[val_index_induc] = True

            features_train = features[train_mask]
            features_val = features[train_val_mask]
            adj_train = utils.adj_preprocess(self.adj,
                                             adj_norm_func=self.adj_norm_func,
                                             mask=self.train_mask,
                                             model_type=model.model_type,
                                             device=self.device)
            adj_val = utils.adj_preprocess(self.adj,
                                           adj_norm_func=self.adj_norm_func,
                                           mask=train_val_mask,
                                           model_type=model.model_type,
                                           device=self.device)

            for epoch in range(n_epoch):
                logits = model(features_train, adj_train, dropout)
                if self.loss == F.nll_loss:
                    out = F.log_softmax(logits, 1)
                    train_loss = self.loss(out, labels[train_mask])
                    logits_val = model(features_val, adj_val, dropout)
                    out_val = F.log_softmax(logits_val, 1)
                    val_loss = self.loss(out_val[val_mask_induc], labels[val_mask])
                elif self.loss == F.cross_entropy:
                    out = logits
                    train_loss = self.loss(out, labels[train_mask])
                    logits_val = model(features_val, adj_val, dropout)
                    out_val = logits_val
                    val_loss = self.loss(out_val[val_mask_induc], labels[val_mask])
                elif self.loss == F.binary_cross_entropy:
                    out = F.sigmoid(logits)
                    train_loss = self.loss(out, labels[train_mask].float())
                    logits_val = model(features_val, adj_val, dropout)
                    out_val = F.sigmoid(logits_val)
                    val_loss = self.loss(out_val[val_mask_induc], labels[val_mask].float())
                elif self.loss == F.binary_cross_entropy_with_logits:
                    out = logits
                    train_loss = self.loss(out, labels[train_mask].float())
                    logits_val = model(features_val, adj_val, dropout)
                    out_val = F.sigmoid(logits_val)
                    val_loss = self.loss(out_val[val_mask_induc], labels[val_mask].float())

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step(val_loss)
                if self.early_stop:
                    self.early_stop(val_loss)
                    if self.early_stop.stop:
                        if verbose:
                            print("Training early stopped.")
                        utils.save_model(model, save_dir, "checkpoint_final.pt", verbose=verbose)
                        return

                if epoch % eval_every == 0:
                    train_score = self.eval_metric(out, labels[train_mask], mask=None)
                    val_score = self.eval_metric(out_val, labels[train_val_mask], mask=val_mask_induc)
                    train_score_list.append(train_score)
                    val_score_list.append(val_score)
                    if val_score > best_val_score:
                        best_val_score = val_score
                        if epoch > save_after:
                            if verbose:
                                print("Epoch {:05d} | Best validation score: {:.4f}".format(epoch, best_val_score))
                            utils.save_model(model, save_dir, save_name, verbose=verbose)
                    if verbose:
                        print(
                            'Epoch {:05d} | Train loss {:.4f} | Train score {:.4f} '
                            '| Val loss {:.4f} | Val score {:.4f}'.format(
                                epoch, train_loss, train_score, val_loss, val_score))
        else:
            # Transductive setting
            adj = utils.adj_preprocess(self.adj,
                                       adj_norm_func=self.adj_norm_func,
                                       mask=None,
                                       model_type=model.model_type,
                                       device=self.device)
            for epoch in range(n_epoch):
                logits = model(features, adj, dropout)
                if self.loss == F.nll_loss:
                    out = F.log_softmax(logits, 1)
                    train_loss = self.loss(out[train_mask], labels[train_mask])
                    val_loss = self.loss(out[val_mask], labels[val_mask])
                elif self.loss == F.cross_entropy:
                    out = logits
                    train_loss = self.loss(out[train_mask], labels[train_mask])
                    val_loss = self.loss(out[val_mask], labels[val_mask])
                elif self.loss == F.binary_cross_entropy:
                    out = F.sigmoid(logits)
                    train_loss = self.loss(out[train_mask], labels[train_mask].float())
                    val_loss = self.loss(out[val_mask], labels[val_mask].float())
                elif self.loss == F.binary_cross_entropy_with_logits:
                    out = logits
                    train_loss = self.loss(out[train_mask], labels[train_mask].float())
                    val_loss = self.loss(out[val_mask], labels[val_mask].float())

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step(val_loss)
                if self.early_stop:
                    self.early_stop(val_loss)
                    if self.early_stop.stop:
                        if verbose:
                            print("Training early stopped.")
                        utils.save_model(model, save_dir, "checkpoint_final.pt", verbose=verbose)
                        return

                if epoch % eval_every == 0:
                    train_score = self.eval_metric(out, labels, train_mask)
                    val_score = self.eval_metric(out, labels, val_mask)
                    train_score_list.append(train_score)
                    val_score_list.append(val_score)
                    if val_score > best_val_score:
                        best_val_score = val_score
                        if epoch > save_after:
                            if verbose:
                                print("Epoch {:05d} | Best validation score: {:.4f}".format(epoch, best_val_score))
                            utils.save_model(model, save_dir, save_name, verbose=verbose)
                    if verbose:
                        print(
                            'Epoch {:05d} | Train loss {:.4f} | Train score {:.4f} '
                            '| Val loss {:.4f} | Val score {:.4f}'.format(
                                epoch, train_loss, train_score, val_loss, val_score))

        utils.save_model(model, save_dir, "checkpoint_final.pt", verbose=verbose)

    def inference(self, model):
        r"""

        Description
        -----------
        Inference of a GNN model.

        Parameters
        ----------
        model : torch.nn.module
            Model implemented based on ``torch.nn.module``.

        Returns
        -------
        logits : torch.Tensor
            Output logits of model.
        test_score : float
            Score on test set.

        """
        model.to(self.device)
        model.eval()
        adj = utils.adj_preprocess(self.adj,
                                   adj_norm_func=self.adj_norm_func,
                                   model_type=model.model_type,
                                   device=self.device)
        logits = model(self.features, adj, dropout=0)
        if self.loss == F.nll_loss:
            out = F.log_softmax(logits, 1)
        elif self.loss == F.binary_cross_entropy:
            out = F.sigmoid(logits)
        else:
            out = logits
        test_score = self.eval_metric(out, self.labels, self.test_mask)

        return logits, test_score


class EarlyStop(object):
    r"""

    Description
    -----------
    Strategy to early stop training process.

    """
    def __init__(self, patience=1000, epsilon=1e-5):
        r"""

        Parameters
        ----------
        patience : int, optional
            Number of epoch to wait if no further improvement. Default: ``1000``.
        epsilon : float, optional
            Tolerance range of improvement. Default: ``1e-4``.

        """
        self.patience = patience
        self.epsilon = epsilon
        self.min_loss = None
        self.stop = False
        self.count = 0

    def __call__(self, loss):
        r"""

        Parameters
        ----------
        loss : float
            Value of loss function.

        """
        if self.min_loss is None:
            self.min_loss = loss
        elif self.min_loss - loss > self.epsilon:
            self.count = 0
            self.min_loss = loss
        elif self.min_loss - loss < self.epsilon:
            self.count += 1
            if self.count > self.patience:
                self.stop = True
