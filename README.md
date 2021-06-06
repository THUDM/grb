# Graph Robustness Benchmark

Graph Robustness Benchmark (GRB) is a scalable, unified, extendable benchmark for evaluating the adversarial robustness of graph neural networks.

## Installation

```
git clone git@github.com:Stanislas0/grb.git
```

## Usage

### Training a GNN model

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

## Requirements

* scipy==1.5.2
* numpy==1.19.1
* torch==1.8.0
* networkx==2.5
* pandas~=1.2.3
* cogdl~=0.3.0.post1
* scikit-learn~=0.24.1

## Contact

If you have any question, please raise an issue or contact qinkai.zheng1028@gmail.com.
