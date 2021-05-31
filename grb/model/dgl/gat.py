import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class GAT(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 num_heads,
                 activation=F.leaky_relu,
                 layer_norm=False):

        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        if layer_norm:
            self.layers.append(nn.LayerNorm(in_features))
        self.layers.append(GATConv(in_features, hidden_features[0], num_heads, activation=activation))
        for i in range(len(hidden_features) - 1):
            if layer_norm:
                self.layers.append(nn.LayerNorm(hidden_features[i] * num_heads))
            self.layers.append(
                GATConv(hidden_features[i] * num_heads, hidden_features[i + 1], num_heads, activation=activation))
        self.layers.append(GATConv(hidden_features[-1] * num_heads, num_heads, out_features))

    @property
    def model_type(self):
        return "dgl"

    def forward(self, x, adj, dropout=0):
        graph = dgl.from_scipy(adj).to(x.device)
        graph.ndata['features'] = x

        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = layer(graph, x).flatten(1)
                if i != len(self.layers) - 1:
                    x = F.dropout(x, dropout)

        return x
