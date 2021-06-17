"""Dataset Module for loading or customizing datasets."""

SUPPORTED_DATASETS = {"grb-cora",
                      "grb-citeseer",
                      "grb-aminer",
                      "grb-reddit",
                      "grb-flickr"}
URLs = {
    "grb-cora"    : {"adj.npz"     : "https://cloud.tsinghua.edu.cn/f/2e522f282e884907a39f/?dl=1",
                     "features.npz": "https://cloud.tsinghua.edu.cn/f/46fd09a8c1d04f11afbb/?dl=1",
                     "labels.npz"  : "https://cloud.tsinghua.edu.cn/f/88fccac46ee94161b48f/?dl=1",
                     "index.npz"   : "https://cloud.tsinghua.edu.cn/f/d8488cbf78a34a8c9c5b/?dl=1"},
    "grb-citeseer": {"adj.npz"     : "https://cloud.tsinghua.edu.cn/f/d3063e4e010e431b95a6/?dl=1",
                     "features.npz": "https://cloud.tsinghua.edu.cn/f/172b66d454d348458bca/?dl=1",
                     "labels.npz"  : "https://cloud.tsinghua.edu.cn/f/f594655156c744da9ef6/?dl=1",
                     "index.npz"   : "https://cloud.tsinghua.edu.cn/f/cb25124f9a454dcf989f/?dl=1"},
    "grb-reddit"  : {"adj.npz"     : "https://cloud.tsinghua.edu.cn/f/22e91d7f34494784a670/?dl=1",
                     "features.npz": "https://cloud.tsinghua.edu.cn/f/000dc5cd8dd643dcbfc6/?dl=1",
                     "labels.npz"  : "https://cloud.tsinghua.edu.cn/f/3e228140ede64b7886b2/?dl=1",
                     "index.npz"   : "https://cloud.tsinghua.edu.cn/f/24310393f5394e3a8b73/?dl=1"},
    "grb-aminer"  : {"adj.npz"     : "https://cloud.tsinghua.edu.cn/f/dca1075cd8cc408bb4c0/?dl=1",
                     "features.npz": "https://cloud.tsinghua.edu.cn/f/e93ba93dbdd94673bce3/?dl=1",
                     "labels.npz"  : "https://cloud.tsinghua.edu.cn/f/0ddbca54864245f3b4e1/?dl=1",
                     "index.npz"   : "https://cloud.tsinghua.edu.cn/f/3444a2e87ef745e89828/?dl=1"},
    "grb-flickr"  : {"adj.npz"     : "https://cloud.tsinghua.edu.cn/f/90a513e35f0a4f3896eb/?dl=1",
                     "features.npz": "https://cloud.tsinghua.edu.cn/f/54b2f1d7ee7c4d5bbcd4/?dl=1",
                     "labels.npz"  : "https://cloud.tsinghua.edu.cn/f/43e9ec09458e4d30b528/?dl=1",
                     "index.npz"   : "https://cloud.tsinghua.edu.cn/f/8239dc6a729e489da44f/?dl=1"},
}

from .dataset import Dataset, CustomDataset, CogDLDataset
