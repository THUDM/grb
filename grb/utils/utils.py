import os
import time
import scipy
import random
import pickle
import torch
import json
import numpy as np
import pandas as pd
from urllib import request

pd.set_option('display.width', 1000)


def adj_to_tensor(adj):
    if type(adj) != scipy.sparse.coo.coo_matrix:
        adj = adj.tocoo()
    sparse_row = torch.LongTensor(adj.row).unsqueeze(1)
    sparse_col = torch.LongTensor(adj.col).unsqueeze(1)
    sparse_concat = torch.cat((sparse_row, sparse_col), 1)
    sparse_data = torch.FloatTensor(adj.data)
    adj_tensor = torch.sparse.FloatTensor(sparse_concat.t(), sparse_data, torch.Size(adj.shape))

    return adj_tensor


def adj_preprocess(adj, adj_norm_func=None, mask=None, model_type="torch", device='cpu'):
    if adj_norm_func is not None:
        adj = adj_norm_func(adj)
    if model_type == "torch":
        if type(adj) is tuple:
            if mask is not None:
                adj = [adj_to_tensor(adj_[mask][:, mask]).to(device) for adj_ in adj]
            else:
                adj = [adj_to_tensor(adj_).to(device) for adj_ in adj]
        else:
            if mask is not None:
                adj = adj_to_tensor(adj[mask][:, mask]).to(device)
            else:
                adj = adj_to_tensor(adj).to(device)
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


def feat_preprocess(features, feat_norm=None, device='cpu'):
    def feat_normalize(feat, norm=None):
        if norm == "arctan":
            feat = 2 * np.arctan(feat) / np.pi
        elif norm == "tanh":
            feat = np.tanh(feat)
        else:
            feat = feat

        return feat

    if type(features) != torch.Tensor:
        features = torch.FloatTensor(features)
    elif features.type() != 'torch.FloatTensor':
        features = features.float()
    if feat_norm is not None:
        features = feat_normalize(features, norm=feat_norm)

    features = features.to(device)

    return features


def label_preprocess(labels, device='cpu'):
    if type(labels) != torch.Tensor:
        labels = torch.LongTensor(labels)
    elif labels.type() != 'torch.LongTensor':
        labels = labels.long()

    labels = labels.to(device)

    return labels


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
            os.makedirs(file_dir)
        np.save(os.path.join(file_dir, file_name), features.cpu().detach().numpy())


def save_adj(adj, file_dir, file_name='adj.pkl'):
    if adj is not None:
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        with open(os.path.join(file_dir, file_name), 'wb') as f:
            pickle.dump(adj, f)


def save_model(model, save_dir, name, verbose=True):
    if save_dir is None:
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        save_dir = "./tmp_{}".format(cur_time)
        os.makedirs(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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


def save_dict_to_xlsx(result_dict, file_dir, file_name="result.xlsx", index=0, verbose=True):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    df = pd.DataFrame(result_dict, index=[index])
    df.to_excel(os.path.join(file_dir, file_name), index=True)
    if verbose:
        print(df)


def save_df_to_xlsx(df, file_dir, file_name="result.xlsx", verbose=False):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    df.to_excel(os.path.join(file_dir, file_name), index=True)
    if verbose:
        print(df)


def save_df_to_csv(df, file_dir, file_name="result.csv", verbose=False):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    df.to_csv(os.path.join(file_dir, file_name), index=True)
    if verbose:
        print(df)


def save_dict_to_json(result_dict, file_dir, file_name, verbose=False):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    with open(os.path.join(file_dir, file_name), 'w') as f:
        json.dump(result_dict, f)
        if verbose:
            print(result_dict)


def check_symmetry(adj):
    if np.sum(adj[:, -adj.shape[0]:].T == adj[:, -adj.shape[0]:]) == adj.shape[0] ** 2:
        return True
    else:
        return False


def check_feat_range(features, feat_lim_min, feat_lim_max):
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if np.min(features) < feat_lim_min or np.max(features) > feat_lim_max:
        return False
    else:
        return True
