import numpy as np
import torch
import torch.nn.functional as F

import grb.utils as utils
from grb.evaluator import metric


class AttackEvaluator(object):
    def __init__(self, dataset, build_model, device="cpu"):
        self.dataset = dataset
        self.device = device
        self.build_model = build_model

    def eval_attack(self, model_dict, adj_attack, features_attack, verbose=False):
        test_score_dict = {}
        for model_name in model_dict.keys():
            model, adj_norm_func = self.build_model(model_name=model_name,
                                                    num_features=self.dataset.num_features,
                                                    num_classes=self.dataset.num_classes)
            model.load_state_dict(torch.load(model_dict[model_name], map_location=self.device))
            model.to(self.device)
            model.eval()

            test_score = self.eval(model=model,
                                   adj=adj_attack,
                                   features=features_attack.to(self.device),
                                   adj_norm_func=adj_norm_func)

            test_score_dict[model_name] = test_score
            if verbose:
                print("Model {}, Test score: {:.4f}".format(model_name, test_score))

        test_score_sorted = sorted(list(test_score_dict.values()))
        test_score_dict["average"] = np.mean(test_score_sorted)
        test_score_dict["3-max"] = np.mean(test_score_sorted[-3:])
        test_score_dict["weighted"] = self.eval_metric(test_score_sorted, metric_type="polynomial")

        return test_score_dict

    def eval(self, model, adj, features, adj_norm_func=None):
        adj_tensor = utils.adj_preprocess(adj=adj,
                                          adj_norm_func=adj_norm_func,
                                          device=self.device,
                                          model_type=model.model_type)
        logits = model(features, adj_tensor, dropout=0)
        logp = F.softmax(logits[:self.dataset.num_nodes], 1)
        test_score = metric.eval_acc(logp, self.dataset.labels.to(self.device),
                                     self.dataset.test_mask.to(self.device))

        return test_score.detach().cpu().numpy()

    @staticmethod
    def eval_metric(test_score_sorted, metric_type="polynomial", order='a'):
        n = len(test_score_sorted)
        if metric_type == "polynomial":
            weights = metric.get_weights_polynomial(n, p=2, order=order)
        elif metric_type == "arithmetic":
            weights = metric.get_weights_arithmetic(n, w_1=0.005, order=order)
        else:
            weights = np.ones(n) / n

        final_score = 0.0
        for i in range(n):
            final_score += test_score_sorted[i] * weights[i]

        return final_score


class DefenseEvaluator(object):
    def __init__(self, dataset, build_model, device="cpu"):
        self.dataset = dataset
        self.device = device
        self.build_model = build_model
#
#     def eval_defense(self, model, attack_dict, adj_attack, features_attack):
#         test_score_dict = {}
#         for attack_name in attack_dict:
#
#         for model_name in model_dict.keys():
#             model, adj_norm_func = self.build_model(model_name=model_name,
#                                                     num_features=self.dataset.num_features,
#                                                     num_classes=self.dataset.num_classes)
#             model.load_state_dict(torch.load(model_dict[model_name]))
#             model.to(self.device)
#             model.eval()
#
#             test_score = self.eval(model=model,
#                                    adj=adj_attack,
#                                    features=features_attack.to(self.device),
#                                    adj_norm_func=adj_norm_func)
#
#             test_score_dict[model_name] = test_score
#
#             print("Model {}, Test score: {:.4f}".format(model_name, test_score))
#
#         test_score_sorted = sorted(list(test_score_dict.values()))
#         test_score_dict["weighted"] = self.eval_metric(test_score_sorted, metric_type="polynomial")
#         test_score_dict["average"] = np.mean(test_score_sorted)
#         test_score_dict["3-max"] = np.mean(test_score_sorted[-3:])
#
#         return test_score_dict
#
#     def eval(self, model, adj, features, adj_norm_func=None):
#         adj_tensor = utils.adj_preprocess(adj, adj_norm_func, self.device)
#         logits = model(features, adj_tensor, dropout=0)
#         logp = F.softmax(logits[:self.dataset.num_nodes], 1)
#         test_score = metric.eval_acc(logp, self.dataset.labels.to(self.device),
#                                      self.dataset.test_mask.to(self.device))
#
#         return test_score.detach().cpu().numpy()
#
#     @staticmethod
#     def eval_metric(test_score_sorted, metric_type="polynomial"):
#         n = len(test_score_sorted)
#         if metric_type == "polynomial":
#             weights = metric.get_weights_polynomial(n, p=2)
#         elif metric_type == "arithmetic":
#             weights = metric.get_weights_arithmetic(n, w_1=0.005)
#         else:
#             weights = np.ones(n) / n
#
#         final_score = 0.0
#         for i in range(n):
#             final_score += test_score_sorted[i] * weights[i]
#
#         return final_score
