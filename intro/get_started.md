# Get Started

<center>
   <img style="border-radius: 0.3125em;"
        width="800"
        src=https://github.com/THUDM/grb/raw/master/docs/source/_static/grb_framework.png>    
  <br>    
  <div style="color:black; 1px solid #d9d9d9;
              font-size: 20px;    
              display: inline-block;
              padding: 2px;">GRB framework. </div> 
  <br>
</center>

GRB is mainly built on [PyTorch](https://pytorch.org/), and also supports popular graph learning libraries like [CogDL](https://github.com/THUDM/cogdl) and [DGL](https://github.com/dmlc/dgl). It provides a modular coding framework, which allows users to conveniently use the implemented methods, and to add new ones. It contains several modules that support the process illustrated in the above figure: (1) *Dataset*: loads GRB datasets and applies necessary preprocessing including splitting scheme and features normalization; it also allows users to customize their own datasets and make them compatible with GRB evaluation framework. (2) *Model*: implements GNN models, which supports models built on pure Pytorch, CogDL or DGL by automatically transforming the inputs to the required formats. (3) *Attack*: builds adversarial attacks on GNNs, the process of attack is abstracted to different components. (4) *Defense*: engages defense mechanism to GNN models, including *preprocess-based* and *model-based* defenses. (5) *Evaluator*: evaluates one/multiple methods under unified evaluation settings, i.e. same datasets, constraints and evaluation metrics. (6) *Pipeline*: unifies the entire process of evaluation: load datasets, train/load models, apply attacks/defenses, and finally get the robustness evaluation results; it also helps to easily reproduce the exact results on GRB leaderboards. Apart from these modules, there are also some others like *Trainer* for model training, *Visualise* for visualizing the attack process. 

This implementation framework allows GRB to have the following features: 

* *Easy-to-use*: GNN models or attacks can be easily built by only a few lines of codes. 
* *Fair-to-compare*: All methods can be fairly compared under unified settings. 
* *Up-to-date*: GRB maintains leaderboards for each dataset and continuously track the progress of this domain. 
* *Guarantee-to-reproduce*: Unlike other benchmarks that just display the results, GRB attaches great importance to reproducibility. For reproducing results on leaderboards, all necessary components are available, including model weights, attack parameters, generated adversarial results, etc. Besides, GRB provides scripts that allow users to reproduce results by a single command line. GRB also provides full [documentation](https://grb.readthedocs.io/en/latest/) for each module and function.

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
