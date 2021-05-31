import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""

    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 learn_eps=True,
                 neighbor_pooling_type='sum',
                 num_mlp_layers=1):
        super(GIN, self).__init__()
        self.learn_eps = learn_eps

        # List of MLPs
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for i in range(len(hidden_features)):
            if i == 0:
                mlp = MLP(num_mlp_layers, in_features, hidden_features[i], hidden_features[i])
            else:
                mlp = MLP(num_mlp_layers, hidden_features[i], hidden_features[i], hidden_features[i])

            self.layers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_features[i]))

        self.linear1 = nn.Linear(hidden_features[-2], hidden_features[-1])
        self.linear2 = nn.Linear(hidden_features[-1], out_features)

    @property
    def model_type(self):
        return "dgl"

    def forward(self, x, adj, dropout=0):
        graph = dgl.from_scipy(adj).to(x.device)
        graph.ndata['features'] = x

        for i in range(len(self.layers) - 1):
            x = self.layers[i](graph, x)
            x = self.batch_norms[i](x)
            x = F.relu(x)

        x = F.relu(self.linear1(x))
        x = F.dropout(x, dropout)
        x = self.linear2(x)

        return x
