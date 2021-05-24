import os
import time
import torch
import torch.nn.functional as F

import grb.utils as utils
from grb.evaluator import metric


class Trainer(object):
    def __init__(self, dataset, optimizer, loss, adj_norm_func=None, train_mode="trasductive", lr_scheduler=None,
                 early_stop=None, device='cpu'):

        # Load dataset
        self.adj = dataset.adj
        self.features = dataset.features
        self.labels = dataset.labels
        self.train_mask = dataset.train_mask
        self.val_mask = dataset.val_mask
        self.test_mask = dataset.test_mask
        self.num_classes = dataset.num_classes
        self.train_mode = train_mode
        self.adj_norm_func = adj_norm_func

        self.device = device
        self.features = torch.FloatTensor(self.features).to(self.device)
        self.labels = torch.LongTensor(self.labels).to(self.device)

        # Settings
        self.optimizer = optimizer
        self.loss = loss

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

    def adj_preprocess(self, adj, adj_norm_func=None, mask=None, model_type="torch"):
        if adj_norm_func is not None:
            adj = adj_norm_func(adj)
        if model_type == "torch":
            if type(adj) is tuple:
                if mask is not None:
                    adj = [utils.adj_to_tensor(adj_[mask][:, mask]).to(self.device) for adj_ in adj]
                else:
                    adj = [utils.adj_to_tensor(adj_).to(self.device) for adj_ in adj]
            else:
                if mask is not None:
                    adj = utils.adj_to_tensor(adj[mask][:, mask]).to(self.device)
                else:
                    adj = utils.adj_to_tensor(adj).to(self.device)
        elif model_type == "dgl":
            if type(adj) is tuple:
                if mask is not None:
                    adj = [adj_[mask][:, mask] for adj_ in adj]
                else:
                    adj = [adj_ for adj_ in adj]
            else:
                if mask is not None:
                    adj = adj[mask][:, mask]
                else:
                    adj = adj
        return adj

    def train(self, model, n_epoch, save_dir=None, save_name=None,
              eval_every=10, save_after=0, dropout=0, verbose=True):
        model.to(self.device)
        model.train()

        if save_dir is None:
            cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            save_dir = "./tmp_{}".format(cur_time)
        else:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

        if save_name is None:
            save_name = "checkpoint.pt"
        else:
            if save_name.split(".")[-1] != "pt":
                save_name = save_name + ".pt"

        train_acc_list = []
        val_acc_list = []
        best_val_acc = 0.0
        features = self.features
        train_mask = self.train_mask
        val_mask = self.val_mask
        labels = self.labels

        if self.train_mode == "inductive":
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
            adj_train = self.adj_preprocess(self.adj,
                                            adj_norm_func=self.adj_norm_func,
                                            mask=self.train_mask,
                                            model_type=model.model_type)
            adj_val = self.adj_preprocess(self.adj,
                                          adj_norm_func=self.adj_norm_func,
                                          mask=train_val_mask,
                                          model_type=model.model_type)

            for epoch in range(n_epoch):
                logits = model(features_train, adj_train, dropout)
                logp = F.log_softmax(logits, 1)
                train_loss = F.nll_loss(logp, labels[train_mask])
                logits_var = model(features_val, adj_val, dropout)
                logp_val = F.log_softmax(logits_var, 1)
                val_loss = F.nll_loss(logp_val[val_mask_induc], labels[val_mask])

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step(val_loss)
                if self.early_stop:
                    self.early_stop(val_loss)
                    if self.early_stop.stop:
                        print("Training early stopped.")
                        utils.save_model(model, save_dir, "checkpoint_final.pt")
                        return

                if epoch % eval_every == 0:
                    train_acc = metric.eval_acc(logp, labels[train_mask], None)
                    val_acc = metric.eval_acc(logp_val, labels[train_val_mask], val_mask_induc)
                    train_acc_list.append(train_acc)
                    val_acc_list.append(val_acc)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        if epoch > save_after:
                            print("Best validation accuracy: {:.4f}".format(best_val_acc))
                            utils.save_model(model, save_dir, save_name)
                    if verbose:
                        print(
                            'Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f}'.format(
                                epoch, train_loss, train_acc, val_loss, val_acc))
        else:
            # Transductive setting
            adj = self.adj_preprocess(self.adj,
                                      adj_norm_func=self.adj_norm_func,
                                      mask=None,
                                      model_type=model.model_type)
            for epoch in range(n_epoch):
                logits = model(features, adj, dropout)
                logp = F.log_softmax(logits, 1)
                train_loss = F.nll_loss(logp[train_mask], labels[train_mask])
                val_loss = F.nll_loss(logp[val_mask], labels[val_mask])

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step(val_loss)
                if self.early_stop:
                    self.early_stop(val_loss)
                    if self.early_stop.stop:
                        print("Training early stopped.")
                        utils.save_model(model, save_dir, "checkpoint_final.pt")
                        return

                if epoch % eval_every == 0:
                    train_acc = metric.eval_acc(logp, labels, train_mask)
                    val_acc = metric.eval_acc(logp, labels, val_mask)
                    train_acc_list.append(train_acc)
                    val_acc_list.append(val_acc)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        if epoch > save_after:
                            print("Best validation accuracy: {:.4f}".format(best_val_acc))
                            utils.save_model(model, save_dir, save_name)
                    if verbose:
                        print(
                            'Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f}'.format(
                                epoch, train_loss, train_acc, val_loss, val_acc))

        utils.save_model(model, save_dir, "checkpoint_final.pt")

    def inference(self, model):
        model.to(self.device)
        model.eval()
        adj = self.adj_preprocess(self.adj,
                                  adj_norm_func=self.adj_norm_func,
                                  model_type=model.model_type)
        logits = model(self.features, adj, dropout=0)
        logp = F.softmax(logits, 1)
        test_acc = metric.eval_acc(logp, self.labels, self.test_mask)

        return logits, test_acc


class EarlyStop(object):
    def __init__(self, patience=1000, epsilon=1e-5):
        self.patience = patience
        self.epsilon = epsilon
        self.min_loss = None
        self.stop = False
        self.count = 0

    def __call__(self, val_loss):
        if self.min_loss is None:
            self.min_loss = val_loss
        elif self.min_loss - val_loss > self.epsilon:
            self.count = 0
            self.min_loss = val_loss
        elif self.min_loss - val_loss < self.epsilon:
            self.count += 1
            if self.count > self.patience:
                self.stop = True
