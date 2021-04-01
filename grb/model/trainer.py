import torch
import torch.nn.functional as F

from grb.utils import utils, evaluator


class Trainer(object):
    def __init__(self, dataset, optimizer, loss, device='cpu', **kwargs):

        # Load dataset
        self.adj = dataset.adj
        self.features = dataset.features
        self.labels = dataset.labels
        self.train_mask = dataset.train_mask
        self.val_mask = dataset.val_mask
        self.test_mask = dataset.test_mask
        self.num_classes = dataset.num_classes

        # Convert to tensor
        self.device = device
        self.prepare()

        # Settings
        self.optimizer = optimizer
        self.loss = loss
        self.config = {}

    def set_config(self, n_epoch, save_path, eval_every=10):
        self.config['n_epoch'] = n_epoch
        self.config['eval_every'] = eval_every
        self.config['save_path'] = save_path

    def prepare(self):
        self.adj = utils.adj_to_tensor(self.adj).to(self.device)
        self.features = torch.FloatTensor(self.features).to(self.device)
        self.labels = torch.LongTensor(self.labels).to(self.device)

    def train(self, model, **kwargs):
        model = model.to(self.device)

        for epoch in range(self.config['n_epoch']):
            logits = model(self.features, self.adj).to(self.device)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[self.train_mask], self.labels[self.train_mask]).to(self.device)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % self.config['eval_every'] == 0:
                train_acc = evaluator.eval_acc(logp, self.labels, self.train_mask)
                val_acc = evaluator.eval_acc(logp, self.labels, self.val_mask)

                print('Epoch {:05d} | Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f}'.format(
                    epoch, loss, train_acc, val_acc))

        torch.save(model.state_dict(), self.config['save_path'])

    def inference(self):
        raise NotImplementedError
