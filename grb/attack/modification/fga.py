import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..base import ModificationAttack
from ...utils import utils


class FGA(ModificationAttack):
    """
    FGA: Fast Gradient Attack on Network Embedding (https://arxiv.org/pdf/1809.02797.pdf)
    """

    def __init__(self,
                 n_edge_mod,
                 loss=F.cross_entropy,
                 allow_isolate=True,
                 device="cpu",
                 verbose=True):
        self.n_edge_mod = n_edge_mod
        self.allow_isolate = allow_isolate
        self.loss = loss
        self.device = device
        self.verbose = verbose

    def attack(self,
               model,
               adj,
               features,
               index_target,
               feat_norm=None,
               adj_norm_func=None):

        features = utils.feat_preprocess(features=features,
                                         feat_norm=model.feat_norm if feat_norm is None else feat_norm,
                                         device=self.device)
        adj_tensor = utils.adj_preprocess(adj=adj,
                                          adj_norm_func=model.adj_norm_func if adj_norm_func is None else adj_norm_func,
                                          model_type=model.model_type,
                                          device=self.device)
        model.to(self.device)
        pred_origin = model(features, adj_tensor)
        labels_origin = torch.argmax(pred_origin, dim=1)

        adj_attack = self.modification(model=model,
                                       adj_origin=adj,
                                       features_origin=features,
                                       labels_origin=labels_origin,
                                       index_target=index_target,
                                       feat_norm=feat_norm,
                                       adj_norm_func=adj_norm_func)

        return adj_attack

    def modification(self,
                     model,
                     adj_origin,
                     features_origin,
                     labels_origin,
                     index_target,
                     feat_norm=None,
                     adj_norm_func=None):
        model.eval()
        adj_attack = adj_origin.todense()
        adj_attack = torch.FloatTensor(adj_attack)
        features_origin = utils.feat_preprocess(features=features_origin,
                                                feat_norm=model.feat_norm if feat_norm is None else feat_norm,
                                                device=self.device)
        adj_attack.requires_grad = True
        n_edge_flip = 0
        for _ in tqdm(range(adj_attack.shape[1])):
            if n_edge_flip >= self.n_edge_mod:
                break
            adj_attack_tensor = utils.adj_preprocess(adj=adj_attack,
                                                     adj_norm_func=model.adj_norm_func if adj_norm_func is None else adj_norm_func,
                                                     model_type=model.model_type,
                                                     device=self.device)
            degs = adj_attack_tensor.sum(dim=1)
            pred = model(features_origin, adj_attack_tensor)
            loss = self.loss(pred[index_target], labels_origin[index_target])
            grad = torch.autograd.grad(loss, adj_attack)[0]
            grad = (grad + grad.T) / torch.Tensor([2.0])
            grad_max = torch.max(grad[index_target], dim=1)
            index_max_i = torch.argmax(grad_max.values)
            index_max_j = grad_max.indices[index_max_i]
            index_max_i = index_target[index_max_i]
            if adj_attack[index_max_i][index_max_j] == 0:
                adj_attack.data[index_max_i][index_max_j] = 1
                adj_attack.data[index_max_j][index_max_i] = 1
                n_edge_flip += 1
            else:
                if self.allow_isolate:
                    adj_attack.data[index_max_i][index_max_j] = 0
                    adj_attack.data[index_max_j][index_max_i] = 0
                    n_edge_flip += 1
                else:
                    if degs[index_max_i] > 1 and degs[index_max_j] > 1:
                        adj_attack.data[index_max_i][index_max_j] = 0
                        adj_attack.data[index_max_j][index_max_i] = 0
                        degs[index_max_i] -= 1
                        degs[index_max_j] -= 1
                        n_edge_flip += 1

        adj_attack = adj_attack.detach().cpu().numpy()
        adj_attack = sp.csr_matrix(adj_attack)
        if self.verbose:
            print("FGA attack finished. {:d} edges were flipped.".format(n_edge_flip))

        return adj_attack
