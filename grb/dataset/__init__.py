import numpy as np

URLs = {
    "grb-cora": {"adj.npz": "https://cloud.tsinghua.edu.cn/f/2e522f282e884907a39f/?dl=1",
                 "features.npz": "https://cloud.tsinghua.edu.cn/f/46fd09a8c1d04f11afbb/?dl=1",
                 "labels.npz": "https://cloud.tsinghua.edu.cn/f/88fccac46ee94161b48f/?dl=1",
                 "index.npz": "https://cloud.tsinghua.edu.cn/f/d8488cbf78a34a8c9c5b/?dl=1"},
    "grb-reddit": {"adj.npz": "https://cloud.tsinghua.edu.cn/f/22e91d7f34494784a670/?dl=1",
                   "features.npz": "https://cloud.tsinghua.edu.cn/f/000dc5cd8dd643dcbfc6/?dl=1",
                   "labels.npz": "https://cloud.tsinghua.edu.cn/f/3e228140ede64b7886b2/?dl=1",
                   "index.npz": "https://cloud.tsinghua.edu.cn/f/24310393f5394e3a8b73/?dl=1"},
    "grb-aminer": {"adj.npz": "https://cloud.tsinghua.edu.cn/f/dca1075cd8cc408bb4c0/?dl=1",
                   "features.npz": "https://cloud.tsinghua.edu.cn/f/e93ba93dbdd94673bce3/?dl=1",
                   "labels.npz": "https://cloud.tsinghua.edu.cn/f/0ddbca54864245f3b4e1/?dl=1",
                   "index.npz": "https://cloud.tsinghua.edu.cn/f/3444a2e87ef745e89828/?dl=1"},
    "grb-amazon": {"adj.npz": "",
                   "features.npz": "",
                   "labels.npz": "",
                   "index.npz": ""},
    "grb-yelp": {"adj.npz": "",
                 "features.npz": "",
                 "labels.npz": "",
                 "index.npz": ""},
    "grb-ppi": {"adj.npz": "",
                "features.npz": "",
                "labels.npz": "",
                "index.npz": ""},
}


def splitting(adj, range_min=(0.0, 0.05), range_max=(0.95, 1.0),
              range_easy=(0.05, 0.35), range_medium=(0.35, 0.65),
              range_hard=(0.65, 0.95), ratio_train=0.6,
              ratio_val=0.1, ratio_test=0.1, seed=42):
    def a_not_in_b(a, b):
        c = []
        for i in a:
            if i not in b:
                c.append(i)

        return np.array(c)

    num_nodes = adj.shape[0]
    degs = adj.getnnz(axis=1)
    print("GRB data splitting...")
    print("    Average degree of all nodes: {:.4f}".format(np.mean(degs)))

    degs_index = np.argsort(degs)
    ind_min = int(len(degs_index) * range_min[1])
    ind_max = int(len(degs_index) * range_max[0])
    print("    Average degree of 5% nodes with small degree: {:.4f}".format(
        np.mean(degs[degs_index[:ind_min]])))
    print("    Average degree of 5% nodes with large degree: {:.4f}".format(
        np.mean(degs[degs_index[ind_max:]])))

    # Sampling 'easy' test nodes
    ind_easy_min = int(len(degs_index) * range_easy[0])
    ind_easy_max = int(len(degs_index) * range_easy[1])
    print("    Average degree of 30% nodes (easy): {:.4f}".format(
        np.mean(degs[degs_index[ind_easy_min:ind_easy_max]])))

    np.random.seed(seed)
    ind_easy_sample = np.random.choice(degs_index[ind_easy_min:ind_easy_max],
                                       int(num_nodes * ratio_test), replace=False)
    print("    Randomly sampled {} nodes".format(ind_easy_sample.shape[0]))

    # Sampling 'medium' test nodes
    ind_medium_min = int(len(degs_index) * range_medium[0])
    ind_medium_max = int(len(degs_index) * range_medium[1])
    print("    Average degree of 30% nodes (medium): {:.4f}".format(
        np.mean(degs[degs_index[ind_medium_min:ind_medium_max]])))

    np.random.seed(seed)
    ind_medium_sample = np.random.choice(degs_index[ind_medium_min:ind_medium_max],
                                         int(num_nodes * ratio_test), replace=False)
    print("    Randomly sampled {} nodes".format(ind_medium_sample.shape[0]))

    # Sampling 'hard' test nodes
    ind_hard_min = int(len(degs_index) * range_hard[0])
    ind_hard_max = int(len(degs_index) * range_hard[1])
    print("    Average degree of 30% nodes (hard): {:.4f}".format(
        np.mean(degs[degs_index[ind_hard_min:ind_hard_max]])))

    np.random.seed(seed)
    ind_hard_sample = np.random.choice(degs_index[ind_hard_min:ind_hard_max],
                                       int(num_nodes * ratio_test), replace=False)
    print("    Randomly sampled {} nodes".format(ind_hard_sample.shape[0]))

    ind_test = np.concatenate([ind_easy_sample,
                               ind_medium_sample,
                               ind_hard_sample])

    # Sampling nodes for training and validation
    ind_rest = a_not_in_b(degs_index, ind_test)
    np.random.seed(seed)
    ind_train = np.random.choice(ind_rest, int(num_nodes * ratio_train), replace=False)
    ind_val = a_not_in_b(ind_rest, ind_train)
    print("    Number of training/validation nodes: {}/{}".format(len(ind_train), len(ind_val)))

    if len(ind_train) + len(ind_val) + len(ind_test) == num_nodes:
        print("    No duplicate.")
    else:
        print("    Find duplicates.")

    index = {"index_train": np.sort(ind_train),
             "index_val": np.sort(ind_val),
             "index_test": np.sort(ind_test),
             "index_test_easy": np.sort(ind_easy_sample),
             "index_test_medium": np.sort(ind_medium_sample),
             "index_test_hard": np.sort(ind_hard_sample)}

    return index
