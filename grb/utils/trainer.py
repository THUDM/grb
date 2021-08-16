import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..evaluator import metric
from ..utils import utils


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
    feat_norm : str, optional
        Type of feature normalization, ['arctan', 'tanh']. Default: ``None``.
    lr_scheduler : bool or instance of torch.optim.lr_scheduler, optional
        Whether to use learning rate scheduler. Default: ``None``.
    lr_patience : int, optional
        Patience of lr_scheduler. Only enabled when ``lr_scheduler is not None``. Default: ``100``.
    lr_factor : float, optional
        Decay factor of lr_scheduler. Only enabled when ``lr_scheduler is not None``. Default: ``0.75``.
    lr_min : float, optional
        Minimum value of learning rate. Only enabled when ``lr_scheduler is not None``. Default: ``0.0``.
    early_stop : bool, optional
        Whether to use early stop.
    early_stop_patience : int, optional
        Patience of early_stop. Only enabled when ``early_stop is not None``. Default: ``100``.
    early_stop_epsilon : float, optional
        Tolerance of early_stop. Only enabled when ``early_stop is not None``. Default: ``1e-5``.
    eval_metric : func of grb.metric, optional
        Evaluation metric, like accuracy or F1 score. Default: ``grb.metric.eval_acc``.
    device : str, optional
        Device used to host data. Default: ``cpu``.

    """

    def __init__(self,
                 dataset,
                 optimizer,
                 loss,
                 feat_norm=None,
                 lr_scheduler=None,
                 lr_patience=100,
                 lr_factor=0.75,
                 lr_min=1e-5,
                 early_stop=None,
                 early_stop_patience=100,
                 early_stop_epsilon=1e-5,
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

        self.device = device
        self.features = utils.feat_preprocess(features=self.raw_features,
                                              feat_norm=feat_norm,
                                              device=self.device)
        self.labels = utils.label_preprocess(labels=self.labels,
                                             device=self.device)

        # Settings
        assert isinstance(optimizer, torch.optim.Optimizer), "Optimizer should be instance of torch.optim.Optimizer."
        self.optimizer = optimizer
        self.loss = loss
        self.eval_metric = eval_metric

        # Learning rate scheduling
        if lr_scheduler:
            if isinstance(lr_scheduler, (torch.optim.lr_scheduler._LRScheduler,
                                         torch.optim.lr_scheduler.ReduceLROnPlateau)):
                self.lr_scheduler = lr_scheduler
            else:
                self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    patience=lr_patience,
                    factor=lr_factor,
                    min_lr=lr_min,
                    verbose=True)
        else:
            self.lr_scheduler = None

        # Early stop
        if early_stop:
            if isinstance(early_stop, EarlyStop):
                self.early_stop = early_stop
            else:
                self.early_stop = EarlyStop(patience=early_stop_patience,
                                            epsilon=early_stop_epsilon)
        else:
            self.early_stop = None

    def train(self,
              model,
              n_epoch,
              save_dir=None,
              save_name=None,
              eval_every=10,
              save_after=0,
              train_mode="transductive",
              return_scores=False,
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
        return_scores : bool, optional
            Whether to return list of scores during training. Default: ``False``.
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
            save_name = "model.pt"
        else:
            if save_name.split(".")[-1] != "pt":
                save_name = save_name + ".pt"

        train_score_list = []
        val_score_list = []
        epoch_bar = tqdm(range(n_epoch))
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
                                             adj_norm_func=model.adj_norm_func,
                                             mask=self.train_mask,
                                             model_type=model.model_type,
                                             device=self.device)
            adj_val = utils.adj_preprocess(self.adj,
                                           adj_norm_func=model.adj_norm_func,
                                           mask=train_val_mask,
                                           model_type=model.model_type,
                                           device=self.device)

            for epoch in epoch_bar:
                train_loss, train_score = self.train_step(model, features_train, adj_train, labels,
                                                          pred_mask=None, labels_mask=train_mask)
                train_score_list.append(train_score)
                if epoch % eval_every == 0:
                    val_loss, val_score = self.eval_step(model, features_val, adj_val, labels,
                                                         pred_mask=val_mask_induc, labels_mask=val_mask)
                    val_score_list.append(val_score)
                    if val_score > best_val_score:
                        best_val_score = val_score
                        if epoch > save_after:
                            if verbose:
                                print("Epoch {:05d} | Best validation score: {:.4f}".format(epoch, best_val_score))
                            utils.save_model(model, save_dir, save_name, verbose=verbose)
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step(val_loss)
                    if self.early_stop is not None:
                        self.early_stop(val_loss)
                        if self.early_stop.stop:
                            if verbose:
                                print("Training early stopped. Best validation score: {:.4f}".format(best_val_score))
                            utils.save_model(model, save_dir, "early_stopped_{}".format(save_name), verbose=verbose)
                            if return_scores:
                                return train_score_list, val_score_list
                            else:
                                return

                    if verbose:
                        epoch_bar.set_description('Epoch {:05d} | Train loss {:.4f} | Train score {:.4f} '
                                                  '| Val loss {:.4f} | Val score {:.4f}'.format(
                            epoch, train_loss, train_score, val_loss, val_score))
        else:
            # Transductive setting
            adj = utils.adj_preprocess(self.adj,
                                       adj_norm_func=model.adj_norm_func,
                                       mask=None,
                                       model_type=model.model_type,
                                       device=self.device)
            for epoch in epoch_bar:
                train_loss, train_score = self.train_step(model, features, adj, labels,
                                                          pred_mask=train_mask, labels_mask=train_mask)
                train_score_list.append(train_score)
                if epoch % eval_every == 0:
                    val_loss, val_score = self.eval_step(model, features, adj, labels,
                                                         pred_mask=val_mask, labels_mask=val_mask)
                    val_score_list.append(val_score)
                    if val_score > best_val_score:
                        best_val_score = val_score
                        if epoch > save_after:
                            if verbose:
                                print("Epoch {:05d} | Best validation score: {:.4f}".format(epoch, best_val_score))
                            utils.save_model(model, save_dir, save_name, verbose=verbose)
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step(val_loss)
                    if self.early_stop is not None:
                        self.early_stop(val_loss)
                        if self.early_stop.stop:
                            if verbose:
                                print("Training early stopped. Best validation score: {:.4f}".format(best_val_score))
                            utils.save_model(model, save_dir, "early_stopped_{}".format(save_name), verbose=verbose)
                            if return_scores:
                                return train_score_list, val_score_list
                            else:
                                return
                    if verbose:
                        epoch_bar.set_description('Epoch {:05d} | Train loss {:.4f} | Train score {:.4f} '
                                                  '| Val loss {:.4f} | Val score {:.4f}'.format(
                            epoch, train_loss, train_score, val_loss, val_score))

        utils.save_model(model, save_dir, "final_{}".format(save_name), verbose=verbose)
        print("Training finished. Best validation score: {:.4f}".format(best_val_score))
        if return_scores:
            return train_score_list, val_score_list
        else:
            return

    def train_step(self, model, features, adj, labels, pred_mask=None, labels_mask=None):
        model.train()
        logits = model(features, adj)
        if self.loss == F.nll_loss:
            pred = F.log_softmax(logits, 1)
        elif self.loss == F.binary_cross_entropy:
            pred = F.sigmoid(logits)
        elif self.loss == F.cross_entropy or self.loss == F.binary_cross_entropy_with_logits:
            pred = logits
        else:
            pred = logits
        if pred_mask is not None:
            pred = pred[pred_mask]
        if labels_mask is not None:
            labels = labels[labels_mask]
        if self.loss == F.binary_cross_entropy or self.loss == F.binary_cross_entropy_with_logits:
            labels = labels.float()
        train_loss = self.loss(pred, labels)
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        train_score = self.eval_metric(pred, labels)

        return train_loss, train_score

    def eval_step(self, model, features, adj, labels, pred_mask=None, labels_mask=None):
        model.eval()
        logits = model(features, adj)
        if self.loss == F.nll_loss:
            pred = F.log_softmax(logits, 1)
        elif self.loss == F.binary_cross_entropy:
            pred = F.sigmoid(logits)
        elif self.loss == F.cross_entropy or self.loss == F.binary_cross_entropy_with_logits:
            pred = logits
        else:
            pred = logits
        if pred_mask is not None:
            pred = pred[pred_mask]
        if labels_mask is not None:
            labels = labels[labels_mask]
        if self.loss == F.binary_cross_entropy or self.loss == F.binary_cross_entropy_with_logits:
            labels = labels.float()
        eval_loss = self.loss(pred, labels)
        eval_score = self.eval_metric(pred, labels)

        return eval_loss, eval_score

        pass

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

        """
        model.to(self.device)
        model.eval()
        adj = utils.adj_preprocess(self.adj,
                                   adj_norm_func=model.adj_norm_func,
                                   model_type=model.model_type,
                                   device=self.device)
        logits = model(self.features, adj)

        return logits

    def evaluate(self, model, mask=None):
        r"""

        Description
        -----------
        Evaluation of a GNN model.

        Parameters
        ----------
        model : torch.nn.module
            Model implemented based on ``torch.nn.module``.
        mask : torch.tensor, optional
            Mask of target nodes.  Default: ``None``.

        Returns
        -------
        score : float
            Score on masked nodes.

        """
        model.to(self.device)
        model.eval()
        adj = utils.adj_preprocess(self.adj,
                                   adj_norm_func=model.adj_norm_func,
                                   model_type=model.model_type,
                                   device=self.device)
        logits = model(self.features, adj)
        score = self.eval_metric(logits, self.labels, mask)

        return score


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
