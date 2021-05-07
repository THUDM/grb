import os
import time
import random
import pickle
import torch
import numpy as np


def adj_to_tensor(adj):
    sparse_row = torch.LongTensor(adj.row).unsqueeze(1)
    sparse_col = torch.LongTensor(adj.col).unsqueeze(1)
    sparse_concat = torch.cat((sparse_row, sparse_col), 1)
    sparse_data = torch.FloatTensor(adj.data)
    adj_tensor = torch.sparse.FloatTensor(sparse_concat.t(), sparse_data, torch.Size(adj.shape))

    return adj_tensor


def adj_preprocess(adj, adj_norm_func=None, device='cpu'):
    if adj_norm_func is not None:
        adj_ = adj_norm_func(adj)
    else:
        adj_ = adj
    if type(adj_) is tuple:
        adj_tensor = [adj_to_tensor(adj_tmp).to(device) for adj_tmp in adj_]
    else:
        adj_tensor = adj_to_tensor(adj_).to(device)

    return adj_tensor


def fix_seed(seed=0):
    """
    Fix random process by a seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_num_params(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def save_features(features, file_dir, file_name='features.npy'):
    if features is not None:
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        np.save(os.path.join(file_dir, file_name), features.cpu().detach().numpy())


def save_adj(adj, file_dir, file_name='adj.pkl'):
    if adj is not None:
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        with open(os.path.join(file_dir, file_name), 'wb') as f:
            pickle.dump(adj, f)


def save_model(model, save_dir, name, verbose=True):
    if save_dir is None:
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        save_dir = "./tmp_{}".format(cur_time)
        os.mkdir(save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    torch.save(model.state_dict(), os.path.join(save_dir, name))

    if verbose:
        print("Model saved in '{}'.".format(os.path.join(save_dir, name)))

#
