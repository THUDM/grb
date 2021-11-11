![GRB](https://github.com/THUDM/grb/blob/master/docs/source/_static/grb_logo.png)

[![PyPi Latest Release](https://badge.fury.io/py/grb.svg)](https://pypi.org/project/grb/)
[![Documentation Status](https://readthedocs.org/projects/grb/badge/?version=latest)](https://grb.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/THUDM/grb)](./LICENSE)

**[Homepage](https://cogdl.ai/grb/home)** | **[Paper](https://arxiv.org/abs/2111.04314)** | **[Datasets](https://cogdl.ai/grb/datasets)** | **[Leaderboard](https://cogdl.ai/grb/leaderboard)** | **[Documentation](https://grb.readthedocs.io/en/latest)**

**Graph Robustness Benchmark (GRB)** provides _scalable_, _unified_, _modular_, and _reproducible_ evaluation on the adversarial robustness of graph machine learning models. GRB has **elaborated datasets**, **unified evaluation pipeline**, **modular coding framework**, and **reproducible leaderboards**, which facilitate the developments of graph adversarial learning, summarizing existing progress and generating insights into future research. 

<div align=center><img width="700" src=https://github.com/THUDM/grb/blob/master/docs/source/_static/grb_key_features.png/></div>

## Updates

* [08/11/2021] The final version of our paper is now available in [arxiv](https://arxiv.org/abs/2111.04314), there is also a representation [video](https://recorder-v3.slideslive.com/?share=55351&s=c61f55db-27d4-4c4e-8844-ad30c0e2776b) for brief introduction of GRB.
* [11/10/2021] GRB is accepted by **NeurIPS 2021 Datasets and Benchmarks Track**! Find [our paper](https://openreview.net/pdf?id=NxWUnvwFV4) in OpenReview.
* [26/09/2021] Add support for graph classification task! See [tutorials](https://github.com/THUDM/grb/blob/master/examples/graph_classification/) in ``examples/``.
* [16/09/2021] Add a [paper list](https://github.com/THUDM/grb/blob/master/paperlist/README.md) of state-of-the-art researches about adversarial robustness in graph machine learning (Keep Updating).
* [27/08/2021] Add support for modification attacks! 7 implementations and tutorials:
    * [Implementations](https://github.com/THUDM/grb/blob/master/grb/attack/modification/)
    * [Modification Attack](https://github.com/THUDM/grb/blob/master/examples/node_classification/modification_attack.ipynb)
* [17/08/2021] Add AutoML function based on [optuna](https://github.com/optuna/optuna) for training models:
    * ``AutoTrainer`` in [grb.trainer.trainer](https://github.com/THUDM/grb/blob/master/grb/utils/trainer.py)
    * Tutorial: [Training models with AutoML](https://github.com/THUDM/grb/blob/master/examples/training_automl.ipynb)
* [14/08/2021] Add tutorials based on jupyter notebook in ``examples/``: 
    * [Load Datasets](https://github.com/THUDM/grb/blob/master/examples/node_classification/load_datasets.ipynb)
    * [Training models](https://github.com/THUDM/grb/blob/master/examples/node_classification/training.ipynb)
    * [Injection Attack](https://github.com/THUDM/grb/blob/master/examples/node_classification/injection_attack.ipynb)
## Get Started

### Installation

Install grb via _pip_ (current version v0.1.0):
```bash
pip install grb
```
Install grb via _git_ (for the newest version):
```bash
git clone git@github.com:THUDM/grb.git
cd grb
pip install -e .
```

### Preparation

GRB provides all necessary components to ensure the reproducibility of evaluation results.
Get datasets from [link](https://cloud.tsinghua.edu.cn/d/c77db90e05e74a5c9b8b/) or download them by running the following script:
```bash
cd ./scripts
sh download_dataset.sh
```
Get attack results (adversarial adjacency matrix and features) from [link](https://cloud.tsinghua.edu.cn/d/94b2ea104c2e457d9667/) or download them by running the following script:
```bash
sh download_attack_results.sh
```
Get saved models (model weights) from [link](https://cloud.tsinghua.edu.cn/d/8b51a6b428464340b368/) or download them by running the following script:
```bash
sh download_saved_models.sh
```

## Usage of GRB Modules

### Training a GML model

An example of training Graph Convolutional Network ([GCN](https://arxiv.org/abs/1609.02907)) on _grb-cora_ dataset. 

```python
import torch  # pytorch backend
from grb.dataset import Dataset
from grb.model.torch import GCN
from grb.trainer.trainer import Trainer

# Load data
dataset = Dataset(name='grb-cora', mode='easy',
                  feat_norm='arctan')
# Build model
model = GCN(in_features=dataset.num_features,
            out_features=dataset.num_classes,
            hidden_features=[64, 64])
# Training
adam = torch.optim.Adam(model.parameters(), lr=0.01)
trainer = Trainer(dataset=dataset, optimizer=adam,
                  loss=torch.nn.functional.nll_loss)
trainer.train(model=model, n_epoch=200, dropout=0.5,
              train_mode='inductive')
```

### Adversarial attack

An example of applying Topological Defective Graph Injection Attack ([TDGIA](https://github.com/THUDM/tdgia)) on trained GCN model.

```python
from grb.attack.injection.tdgia import TDGIA

# Attack configuration
tdgia = TDGIA(lr=0.01, 
              n_epoch=10,
              n_inject_max=20, 
              n_edge_max=20,
              feat_lim_min=-0.9, 
              feat_lim_max=0.9,
              sequential_step=0.2)
# Apply attack
rst = tdgia.attack(model=model,
                   adj=dataset.adj,
                   features=dataset.features,
                   target_mask=dataset.test_mask)
# Get modified adj and features
adj_attack, features_attack = rst
```

## GRB Evaluation

### Evaluation scenarios (Injection attack as an example)

![GRB](https://github.com/THUDM/grb/blob/master/docs/source/_static/grb_scenario.png)

GRB provides unified evaluation scenarios for fair comparisons between attacks and defenses. The example scenario is **Black-box**, **Evasion**, **Inductive**, **Injection**. 

* **Black-box**: Both the attacker and the defender have no knowledge about the applied methods each other uses.
* **Evasion**: Models are already trained in trusted data (e.g. authenticated users), which are untouched by the attackers but might have natural noises. Thus, attacks will only happen during the inference phase. 
* **Inductive**: Models are used to classify unseen data (e.g. new users), i.e. validation or test data are unseen during training, which requires models to generalize to out of distribution data.
* **Injection**: The attackers can only inject new nodes but not modify the target nodes directly. Since it is usually hard to hack into users' accounts and modify their profiles. However, it is easier to create fake accounts and connect them to existing users.

### GRB Leaderboards

GRB maintains [leaderboards](https://cogdl.ai/grb/leaderboard/) that permits a fair comparision across various attacks and defenses. To ensure the reproducibility, we provide all necessary information including datasets, attack results, saved models, etc. Besides, all results on the leaderboards can be easily reproduced by running the following scripts (e.g., [leaderboard for _grb-cora_ dataset](https://cogdl.ai/grb/leaderboard/cora, compatible with v0.1.0)):
```bash
sh run_leaderboard_pipeline.sh -d grb-cora -g 0 -s ./leaderboard -n 0
Usage: run_leaderboard_pipeline.sh [-d <string>] [-g <int>] [-s <string>] [-n <int>]
Pipeline for reproducing leaderboard on the chosen dataset.
    -h      Display help message.
    -d      Choose a dataset.
    -s      Set a directory to save leaderboard files.
    -n      Choose the number of an attack from 0 to 9.
    -g      Choose a GPU device. -1 for CPU.
```

## Submission

We welcome researchers to submit new methods including attacks, defenses, or new GML models to enrich the GRB leaderboard. For future submissions, one should follow the [GRB Evaluation Rules](https://cogdl.ai/grb/intro/rules) and respect the reproducibility. 

Please submit your methods via the google form [GRB submission](https://docs.google.com/forms/d/e/1FAIpQLSfJaUK-SXYFnlSqTEEwTOwsqzA5JnpXyvZe8E24hlLE7scRcA/viewform). Our team will verify the result within a week.

## Requirements

* scipy==1.5.2
* numpy==1.19.1
* torch==1.8.0
* networkx==2.5
* pandas~=1.2.3
* cogdl~=0.3.0.post1
* scikit-learn~=0.24.1

## Citing GRB

If you find GRB useful for your research, please cite [our paper](https://arxiv.org/abs/2111.04314):  

```
@article{zheng2021grb,
  title={Graph Robustness Benchmark: Benchmarking the Adversarial Robustness of Graph Machine Learning},
  author={Zheng, Qinkai and Zou, Xu and Dong, Yuxiao and Cen, Yukuo and Yin, Da and Xu, Jiarong and Yang, Yang and Tang, Jie},
  journal={Neural Information Processing Systems Track on Datasets and Benchmarks 2021},
  year={2021}
}
```

## Contact

In case of any problem, please contact us via email: cogdl.grbteam@gmail.com. We also welcome researchers to join our [Google Group](https://groups.google.com/g/graph-robustness-benchmark) for further discussion on the adversarial robustness of graph machine learning.
