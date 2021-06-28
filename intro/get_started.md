# Get Started

**Graph Robustness Benchmark (GRB)** provides _scalable_, _general_, _unified_, and _reproducible_ evaluation on the *adversarial robustness* of graph machine learning models, especially Graph Neural Networks (GNNs). GRB has elaborated datasets, unified evaluation pipeline, modular coding framework, and reproducible leaderboards, which facilitates a fair comparison among various attacks & defenses on GNNs and will promote future research in this field.

<div align=center><img width="700" src=https://github.com/THUDM/grb/blob/master/docs/source/_static/grb_framework.png/></div>

## Installation

Install grb via _pip_:

```bash
pip install grb
```

Install grb via _git_:

```bash
git clone git@github.com:THUDM/grb.git
cd grb
pip install -e .
```

## Usage

### Training a GNN model

An example of training Graph Convolutional Network ([GCN](https://arxiv.org/abs/1609.02907)) on _grb-cora_ dataset.

```python
import torch  # pytorch backend
from grb.dataset import Dataset
from grb.model.torch import GCN
from grb.utils.trainer import Trainer

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
from grb.attack.tdgia import TDGIA

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

### Defense

An example of applying adversarial training on GCN model.

```python
import torch  # pytorch backend
from grb.dataset import Dataset
from grb.model.torch import GCN
from grb.utils.normalize import GCNAdjNorm
from grb.attack import FGSM
from grb.defense import AdvTrainer

# Load data
dataset = Dataset(name='grb-cora', mode='easy',
                  feat_norm='arctan')
# Build model
model = GCN(in_features=dataset.num_features,
            out_features=dataset.num_classes,
            hidden_features=[64, 64])
# Adversarial Training
adam = torch.optim.Adam(model.parameters(), lr=0.01)
advtrainer = AdvTrainer(dataset=dataset, optimizer=adam,
                        adj_norm_func=GCNAdjNorm,
                  			loss=torch.nn.functional.nll_loss)
advtrainer.train(model=model, n_epoch=200, dropout=0.5,
              	 train_mode='inductive')
```

For more information of each module, please refer to [GRB documentation](https://grb.readthedocs.io/en/latest/).
