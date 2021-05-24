import os
import time
import scipy
import random
import pickle
import torch
import numpy as np
from urllib import request


def adj_to_tensor(adj):
    if type(adj) != scipy.sparse.coo.coo_matrix:
        adj = adj.tocoo()
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


def get_index_induc(index_a, index_b):
    i_a, i_b = 0, 0
    l_a, l_b = len(index_a), len(index_b)
    i_new = 0
    index_a_new, index_b_new = [], []
    while i_new < l_a + l_b:
        if i_a == l_a:
            while i_b < l_b:
                i_b += 1
                index_b_new.append(i_new)
                i_new += 1
            continue
        elif i_b == l_b:
            while i_a < l_a:
                i_a += 1
                index_a_new.append(i_new)
                i_new += 1
            continue
        if index_a[i_a] < index_b[i_b]:
            i_a += 1
            index_a_new.append(i_new)
            i_new += 1
        else:
            i_b += 1
            index_b_new.append(i_new)
            i_new += 1

    return index_a_new, index_b_new


def download(url, save_path):
    print("Downloading from {}".format(url))
    try:
        data = request.urlopen(url)
    except Exception as e:
        print(e)
        print("Failed to download the dataset.")
        exit(1)
    with open(save_path, "wb") as f:
        f.write(data.read())
